from copy import deepcopy
from typing import List, Union

from ..dtypes import Integer
from ..representation import Node
from ..tracing import Tracer
from ..values import EncryptedTensor
from .dtypes import TFHERSIntegerType
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
    return _eval_from_native(value)


def _trace_to_native(tfhers_int: Tracer, dtype: TFHERSIntegerType):
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
        attributes={"type": dtype},
    )
    return Tracer(
        computation,
        input_tracers=[
            tfhers_int,
        ],
    )


def _trace_from_native(native_int: Tracer, dtype_to: TFHERSIntegerType):
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
        attributes={"type": dtype_to},
    )
    return Tracer(
        computation,
        input_tracers=[
            native_int,
        ],
    )


def _eval_to_native(tfhers_int: TFHERSInteger):
    return tfhers_int.value


def _eval_from_native(native_value):
    return native_value
