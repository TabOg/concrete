"""
Tests of execution of dynamic indexing operation.
"""

import random

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "encryption_status,function,inputset",
    [
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3], lambda _: np.random.randint(0, 3)),
            id="x[y] where x.shape == (3,) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3], lambda _: np.random.randint(-3, 3)),
            id="x[y] where x.shape == (3,) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 31], lambda _: np.random.randint(0, 3)),
            id="x[y] where x.shape == (31,) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y],
            fhe.inputset(fhe.tensor[fhe.uint4, 31], lambda _: np.random.randint(-3, 3)),
            id="x[y] where x.shape == (31,) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 0],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 3)),
            id="x[y, 0] where x.shape == (3, 4) | 0 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[y, 0],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-3, 3)),
            id="x[y, 0] where x.shape == (3, 4) | -3 < y < 3",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(0, 4)),
            id="x[1, y] where x.shape == (3,, 4) | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear"},
            lambda x, y: x[1, y],
            fhe.inputset(fhe.tensor[fhe.uint4, 3, 4], lambda _: np.random.randint(-4, 4)),
            id="x[1, y] where x.shape == (3,, 4) | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | 0 < y < 3 | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(0, 3),
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | 0 < y < 3 | -4 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(-3, 3),
                lambda _: np.random.randint(0, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | -3 < y < 3 | 0 < y < 4",
        ),
        pytest.param(
            {"x": "encrypted", "y": "clear", "z": "clear"},
            lambda x, y, z: x[y, z],
            fhe.inputset(
                fhe.tensor[fhe.uint4, 3, 4],
                lambda _: np.random.randint(-3, 3),
                lambda _: np.random.randint(-4, 4),
            ),
            id="x[y, z] where x.shape == (3, 4) | -3 < y < 3 | -4 < y < 4",
        ),
    ],
)
def test_dynamic_indexing(encryption_status, function, inputset, helpers):
    """
    Test dynamic indexing.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, encryption_status)
    circuit = compiler.compile(inputset, configuration)

    for sample in random.sample(inputset, 8):
        helpers.check_execution(circuit, function, list(sample))
