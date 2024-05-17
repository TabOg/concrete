from functools import partial
from typing import List, Union

import numpy as np

from .dtypes import TFHERSIntegerType, int8_2_2, int16_2_2, uint8_2_2, uint16_2_2


class TFHERSInteger:
    _value: Union[int, np.ndarray]
    _dtype: TFHERSIntegerType
    _shape: tuple

    def __init__(
        self,
        dtype: TFHERSIntegerType,
        value: Union[List, int, np.ndarray],
    ):
        if isinstance(value, int):
            self._shape = ()
        elif isinstance(value, list):
            try:
                value = np.array(value)
            except Exception as e:  # pylint: disable=broad-except
                msg = f"got error while trying to convert list value into a numpy array: {e}"
                raise ValueError()
        elif isinstance(value, np.ndarray):
            if value.max() > dtype.max():
                raise ValueError(
                    "ndarray value has bigger elements than what the dtype can support"
                )
            if value.min() < dtype.min():
                raise ValueError(
                    "ndarray value has smaller elements than what the dtype can support"
                )
            self._shape = value.shape
        else:
            raise TypeError("value can either be an int or ndarray")

        self._value = value
        self._dtype = dtype

    @property
    def dtype(self) -> TFHERSIntegerType:
        return self._dtype

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def value(self) -> Union[int, np.ndarray]:
        return self._value

    def min(self):
        return self.dtype.min()

    def max(self):
        return self.dtype.max()


int8_2_2_value = partial(TFHERSInteger, int8_2_2)
int16_2_2_value = partial(TFHERSInteger, int16_2_2)
uint8_2_2_value = partial(TFHERSInteger, uint8_2_2)
uint16_2_2_value = partial(TFHERSInteger, uint16_2_2)
