from copy import deepcopy
from typing import List, Union

from ..dtypes import Integer
from ..representation import Node
from ..tracing import Tracer
from ..values import EncryptedScalar, EncryptedTensor
from .dtypes import TFHERSIntegerType, uint16
from .values import TFHERSInteger


def to_native(value: Union[Tracer, TFHERSInteger]):
    if isinstance(value, Tracer):
        dtype = value.output.dtype
        if not isinstance(dtype, TFHERSIntegerType):
            raise TypeError("tracer didn't contain an output of TFHEInteger type. Type is:", dtype)
        return _trace_to_native(value, dtype)
    assert isinstance(value, TFHERSInteger)
    dtype = value.dtype
    return _eval_to_native(value)


def from_native(value, dtype_to: TFHERSIntegerType):
    if isinstance(value, Tracer):
        return _trace_from_native(value, dtype_to)
    return _eval_from_native(value, dtype_to.pad_width, dtype_to.msg_width)


def _trace_to_native(tfhers_int: Tracer, dtype: TFHERSIntegerType):
    if tfhers_int.output.is_scalar:
        output = EncryptedScalar(Integer(dtype.is_signed, dtype.bit_width))
    else:
        output = EncryptedTensor(
            Integer(dtype.is_signed, dtype.bit_width),
            tfhers_int.output.shape,
        )

    computation = Node.generic(
        "tfhers_to_native",
        deepcopy(
            [
                tfhers_int.output,
            ]
        ),
        output,
        _eval_to_native,
        args=(),
        attributes={
            "result_bit_width": dtype.bit_width,
            "pad_width": dtype.pad_width,
            "msg_width": dtype.msg_width,
        },
    )
    return Tracer(
        computation,
        input_tracers=[
            tfhers_int,
        ],
    )


def _trace_from_native(native_int: Tracer, dtype_to: TFHERSIntegerType):
    if native_int.output.is_scalar:
        output = EncryptedScalar(dtype_to)
    else:
        output = EncryptedTensor(dtype_to, native_int.output.shape)

    computation = Node.generic(
        "tfhers_from_native",
        deepcopy(
            [
                native_int.output,
            ]
        ),
        output,
        _eval_from_native,
        args=(),
        kwargs={"pad_width": dtype_to.pad_width, "msg_width": dtype_to.msg_width},
    )
    return Tracer(
        computation,
        input_tracers=[
            native_int,
        ],
    )


def _encode_tfhers_int(value: int, bit_width: int, msg_width: int) -> List[int]:
    if bit_width % msg_width != 0:
        raise ValueError(f"bit_width ({bit_width}) must be a multiple of msg_width ({msg_width})")
    result_size = bit_width // msg_width
    # we remove the leading 0b and fill msb with 0
    value_bin = bin(value)[2:].zfill(bit_width)
    assert len(value_bin) <= bit_width
    # msb first
    values = [int(value_bin[i : i + msg_width], 2) for i in range(0, len(value_bin), msg_width)]
    return values


def _eval_to_native(tfhers_int: TFHERSInteger):
    # dtype = tfhers_int.dtype
    # value = tfhers_int.value
    # if isinstance(value, int):
    #     return _encode_tfhers_int(value, dtype.bit_width, dtype.msg_width)
    # # TODO: ndarray case
    return tfhers_int.value


def _eval_from_native(native_value, pad_width: int, msg_width: int):
    return native_value
