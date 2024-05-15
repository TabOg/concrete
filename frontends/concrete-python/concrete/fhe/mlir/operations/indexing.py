"""
Conversion of indexing operation.
"""

# pylint: disable=import-error,no-name-in-module

from typing import Sequence, Union

import numpy as np
from concrete.lang.dialects import fhelinalg, tracing
from mlir.dialects import arith, scf, tensor
from mlir.ir import ArrayAttr as MlirArrayAttr
from mlir.ir import DenseI64ArrayAttr as MlirDenseI64ArrayAttr
from mlir.ir import IntegerAttr as MlirIntegerAttr
from mlir.ir import ShapedType as MlirShapedType

from ...dtypes import Integer
from ...internal.utils import unreachable
from ...values import ValueDescription
from ..context import Context
from ..conversion import Conversion, ConversionType

# pylint: enable=import-error,no-name-in-module


def process_indexing_element(
    ctx: Context,
    indexing_element: Union[int, np.integer, slice, np.ndarray, list, Conversion],
    dimension_size: int,
) -> Union[int, np.integer, slice, np.ndarray, list, Conversion]:
    if isinstance(indexing_element, (int, np.integer)):
        result = int(indexing_element)
        if result < 0:
            result += dimension_size

        assert 0 <= result < dimension_size
        return result

    elif isinstance(indexing_element, Conversion):
        if indexing_element.is_tensor:
            raise NotImplementedError

        indexing_element_node = indexing_element.origin
        assert isinstance(indexing_element_node.output.dtype, Integer)

        dimension_size_variable: Optional[Conversion] = None
        dimension_size_variable_type = ctx.i(
            max(
                Integer.that_can_represent(dimension_size).bit_width + 1,
                indexing_element.bit_width,
            )
        )

        if indexing_element_node.output.dtype.is_signed:
            dimension_size_variable = ctx.constant(dimension_size_variable_type, dimension_size)
            if indexing_element.bit_width < dimension_size_variable.bit_width:
                indexing_element = ctx.operation(
                    arith.ExtSIOp,
                    ctx.tensor(dimension_size_variable_type, shape=indexing_element.shape),
                    indexing_element.result,
                )

            offset_condition = ctx.operation(
                arith.CmpIOp,
                ctx.i(1),
                ctx.attribute(ctx.i(64), 2),  # signed less than
                indexing_element.result,
                ctx.constant(indexing_element.type, 0).result,
            )

            indexing_element = ctx.conditional(
                indexing_element.type,
                offset_condition,
                lambda: ctx.operation(
                    arith.AddIOp,
                    indexing_element.type,
                    indexing_element.result,
                    dimension_size_variable.result,
                ),
                lambda: indexing_element,
            )

        if ctx.configuration.dynamic_indexing_check_out_of_bound:
            dimension_size_variable = ctx.constant(dimension_size_variable_type, dimension_size)
            if indexing_element.bit_width < dimension_size_variable.bit_width:
                indexing_element = ctx.operation(
                    arith.ExtSIOp,
                    ctx.tensor(dimension_size_variable_type, shape=indexing_element.shape),
                    indexing_element.result,
                )

            def warn_out_of_memory_access():
                pred_ids = [
                    pred.properties["id"] for pred in ctx.graph.ordered_preds_of(ctx.converting)
                ]
                operation_string = ctx.converting.format(pred_ids)
                tracing.TraceMessageOp(
                    msg=(
                        f"Runtime Warning: Index out of range on "
                        f"\"{ctx.converting.properties['id']} = {operation_string}\"\n"
                    )
                )

            indexing_element_too_large_condition = ctx.operation(
                arith.CmpIOp,
                ctx.i(1),
                ctx.attribute(ctx.i(64), 5),  # signed greater than or equal
                indexing_element.result,
                dimension_size_variable.result,
            )

            ctx.conditional(
                None,
                indexing_element_too_large_condition,
                warn_out_of_memory_access,
            )

            indexing_element_too_small_condition = ctx.operation(
                arith.CmpIOp,
                ctx.i(1),
                ctx.attribute(ctx.i(64), 2),  # signed less than
                indexing_element.result,
                ctx.constant(indexing_element.type, 0).result,
            )

            ctx.conditional(
                None,
                indexing_element_too_small_condition,
                warn_out_of_memory_access,
            )

        return ctx.operation(
            arith.IndexCastOp,
            ctx.tensor(ctx.index_type(), shape=indexing_element.shape),
            indexing_element.result,
        )

    else:
        unreachable()


def indexing(
    ctx: Context,
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, np.integer, slice, np.ndarray, list, Conversion]],
) -> Conversion:
    assert resulting_type.is_encrypted == x.is_encrypted
    assert ctx.is_bit_width_compatible(resulting_type, x)

    is_fancy = any(
        (
            isinstance(indexing_element, (list, np.ndarray))
            or (isinstance(indexing_element, Conversion) and indexing_element.is_tensor)
        )
        for indexing_element in index
    )
    if is_fancy:
        indices_shape = resulting_type.shape

        processed_indices = []
        for indexing_element in index:
            if isinstance(indexing_element, Conversion):
                # TODO: check for out of bounds
                processed_index = ctx.operation(
                    arith.IndexCastOp,
                    ctx.tensor(ctx.index_type(), shape=indexing_element.shape),
                    indexing_element.result,
                )
                processed_indices.append(processed_index)
                continue

            if isinstance(indexing_element, (int, np.integer)):
                processed_index = ctx.constant(ctx.index_type(), indexing_element)
                processed_indices.append(processed_index)
                continue

            if isinstance(indexing_element, (list, np.ndarray)):
                indexing_element = np.array(indexing_element)
                processed_index = ctx.constant(ctx.tensor(ctx.index_type(), indexing_element.shape), indexing_element)
                processed_indices.append(processed_index)
                continue

            # pragma: no cover
            message = f"invalid indexing element of type {type(indexing_element)}"
            raise AssertionError(message)

        if len(processed_indices) == 1:
            assert len(x.shape) == 1
            indices = processed_indices[0]
        else:
            indices = ctx.concatenate(
                ctx.tensor(ctx.index_type(), indices_shape + (len(processed_indices),)),
                [
                    ctx.reshape(ctx.broadcast_to(index, indices_shape), shape=(indices_shape + (1,)))
                    for index in processed_indices
                ],
                axis=-1,
            )

        return ctx.operation(
            fhelinalg.FancyIndexOp,
            resulting_type,
            x.result,
            indices.result,
        )

    index = list(index)
    while len(index) < len(x.shape):
        index.append(slice(None, None, None))

    if resulting_type.shape == ():
        processed_index = []
        for indexing_element, dimension_size in zip(index, x.shape):
            assert isinstance(indexing_element, (int, np.integer, Conversion))

            processed_indexing_element = process_indexing_element(
                ctx,
                indexing_element,
                dimension_size,
            )
            if not isinstance(processed_indexing_element, Conversion):
                processed_indexing_element = ctx.constant(
                    ctx.index_type(),
                    processed_indexing_element,
                )

            processed_index.append(processed_indexing_element)

        return ctx.operation(
            tensor.ExtractOp,
            resulting_type,
            x.result,
            tuple(
                processed_indexing_element.result for processed_indexing_element in processed_index
            ),
            original_bit_width=x.original_bit_width,
        )

    dynamic_offsets = []
    dynamic_sizes = []
    dynamic_strides = []

    static_offsets = []
    static_sizes = []
    static_strides = []

    destroyed_dimensions = []
    for dimension, (indexing_element, dimension_size) in enumerate(zip(index, x.shape)):
        if isinstance(indexing_element, slice):
            size = int(np.zeros(dimension_size)[indexing_element].shape[0])
            stride = int(indexing_element.step if indexing_element.step is not None else 1)
            offset = int(
                process_indexing_element(ctx, indexing_element.start, dimension_size)
                if indexing_element.start is not None
                else (0 if stride > 0 else dimension_size - 1)
            )
        else:
            assert isinstance(indexing_element, (int, np.integer)) or (
                isinstance(indexing_element, Conversion) and indexing_element.is_scalar
            )
            destroyed_dimensions.append(dimension)

            size = 1
            stride = 1
            offset = process_indexing_element(ctx, indexing_element, dimension_size)

            if isinstance(offset, Conversion):
                dynamic_offsets.append(offset)
                offset = MlirShapedType.get_dynamic_size()

        static_offsets.append(offset)
        static_sizes.append(size)
        static_strides.append(stride)

    if len(destroyed_dimensions) == 0:
        return ctx.operation(
            tensor.ExtractSliceOp,
            resulting_type,
            x.result,
            tuple(map(lambda item: item.result, dynamic_offsets)),
            tuple(map(lambda item: item.result, dynamic_sizes)),
            tuple(map(lambda item: item.result, dynamic_strides)),
            MlirDenseI64ArrayAttr.get(static_offsets),
            MlirDenseI64ArrayAttr.get(static_sizes),
            MlirDenseI64ArrayAttr.get(static_strides),
            original_bit_width=x.original_bit_width,
        )

    intermediate_shape = list(resulting_type.shape)
    for dimension in destroyed_dimensions:
        intermediate_shape.insert(dimension, 1)

    intermediate_type = ctx.typeof(
        ValueDescription(
            dtype=Integer(is_signed=x.is_signed, bit_width=x.bit_width),
            shape=tuple(intermediate_shape),
            is_encrypted=x.is_encrypted,
        )
    )

    intermediate = ctx.operation(
        tensor.ExtractSliceOp,
        intermediate_type,
        x.result,
        tuple(map(lambda item: item.result, dynamic_offsets)),
        tuple(map(lambda item: item.result, dynamic_sizes)),
        tuple(map(lambda item: item.result, dynamic_strides)),
        MlirDenseI64ArrayAttr.get(static_offsets),
        MlirDenseI64ArrayAttr.get(static_sizes),
        MlirDenseI64ArrayAttr.get(static_strides),
    )

    reassociaton = []

    current_intermediate_dimension = 0
    for _ in range(len(resulting_type.shape)):
        indices = [current_intermediate_dimension]
        while current_intermediate_dimension in destroyed_dimensions:
            current_intermediate_dimension += 1
            indices.append(current_intermediate_dimension)

        reassociaton.append(indices)
        current_intermediate_dimension += 1
    while current_intermediate_dimension < len(intermediate_shape):
        reassociaton[-1].append(current_intermediate_dimension)
        current_intermediate_dimension += 1

    return ctx.operation(
        tensor.CollapseShapeOp,
        resulting_type,
        intermediate.result,
        MlirArrayAttr.get(
            [
                MlirArrayAttr.get(
                    [MlirIntegerAttr.get(ctx.i(64).mlir, index) for index in indices],
                )
                for indices in reassociaton
            ],
        ),
    )
