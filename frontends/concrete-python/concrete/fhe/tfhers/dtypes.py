"""
Declaration of `TFHERSIntegerType` class.
"""

from functools import partial
from typing import Any

from ..dtypes import Integer


class TFHERSIntegerType(Integer):
    pad_width: int
    msg_width: int

    def __init__(self, is_signed: bool, bit_width: int, pad_width: int, msg_width: int):
        super().__init__(is_signed, bit_width)
        self.pad_width = pad_width
        self.msg_width = msg_width

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and super().__eq__(other)
            and self.pad_width == other.pad_width
            and self.msg_width == other.msg_width
        )

    def __str__(self) -> str:
        return (
            f"tfhers_{('int' if self.is_signed else 'uint')}"
            f"{self.bit_width}_{self.pad_width}_{self.msg_width}"
        )


int8 = partial(TFHERSIntegerType, True, 8)
uint8 = partial(TFHERSIntegerType, False, 8)
int16 = partial(TFHERSIntegerType, True, 16)
uint16 = partial(TFHERSIntegerType, False, 16)

int8_2_2 = int8(2, 2)
uint8_2_2 = uint8(2, 2)
int16_2_2 = int16(2, 2)
uint16_2_2 = uint16(2, 2)
