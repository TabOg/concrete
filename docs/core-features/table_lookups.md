# Table lookups
This document explains the use of table lookups (TLUs) in Concrete, covering creation, exactness, and performance, and the configuration options to manage error probabilities and improve efficiency.

Table lookups (TLUs) are one of the most common operations in **Concrete**. Most operations are converted to TLUs under the hood, except:
- Addition
- Subtraction
- Multiplication with non-encrypted values
- Tensor manipulation
- Other operations built from these primitives, such as matmul, conv. and so on.

Table lookups are very flexible, enabling **Concrete** to support a wide range of operations. However, TLUs are always much more expensive than other operations, even though the exact cost depends on many variables, such as hardware used and error probability.

Therefore, when feasible, you should reducing the number of TLUs or replace some of them with other primitive operations.

{% hint style="info" %}
**Concrete** automatically parallelizes TLUs when applied to tensors.
{% endhint %}

## Direct table lookup

**Concrete** provides a `LookupTable` class that allows you to create your own tables and apply them in your circuits.

{% hint style="info" %}
`LookupTable` can have any number of elements, denoted as  **N**. The table lookup is valid if the lookup variable is within the range [-N, N). 

If the lookup variable is outside this range, you will receive the following error:

```
IndexError: index 10 is out of bounds for axis 0 with size 6
```
{% endhint %}

### With scalars

You can create the lookup table using a list of integers and apply it using indexing:

```python
from concrete import fhe

table = fhe.LookupTable([2, -1, 3, 0])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = range(4)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0) == table[0] == 2
assert circuit.encrypt_run_decrypt(1) == table[1] == -1
assert circuit.encrypt_run_decrypt(2) == table[2] == 3
assert circuit.encrypt_run_decrypt(3) == table[3] == 0
```

### With tensors

When you apply a table lookup to a tensor, the scalar table lookup is applied to each element of the tensor:

```python
from concrete import fhe
import numpy as np

table = fhe.LookupTable([2, -1, 3, 0])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = [np.random.randint(0, 4, size=(2, 3)) for _ in range(10)]
circuit = f.compile(inputset)

sample = [
    [0, 1, 3],
    [2, 3, 1],
]
expected_output = [
    [2, -1, 0],
    [3, 0, -1],
]
actual_output = circuit.encrypt_run_decrypt(np.array(sample))

for i in range(2):
    for j in range(3):
        assert actual_output[i][j] == expected_output[i][j] == table[sample[i][j]]
```

### With negative values

`LookupTable` mimics array indexing in Python, which means if the lookup variable is negative, the table is looked up from the back:

```python
from concrete import fhe

table = fhe.LookupTable([2, -1, 3, 0])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[-x]

inputset = range(1, 5)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(1) == table[-1] == 0
assert circuit.encrypt_run_decrypt(2) == table[-2] == 3
assert circuit.encrypt_run_decrypt(3) == table[-3] == -1
assert circuit.encrypt_run_decrypt(4) == table[-4] == 2
```

## Direct multi-table lookup

To apply a different lookup table to each element of a tensor, you can have a `LookupTable` of `LookupTable`s:

```python
from concrete import fhe
import numpy as np

squared = fhe.LookupTable([i ** 2 for i in range(4)])
cubed = fhe.LookupTable([i ** 3 for i in range(4)])

table = fhe.LookupTable([
    [squared, cubed],
    [squared, cubed],
    [squared, cubed],
])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = [np.random.randint(0, 4, size=(3, 2)) for _ in range(10)]
circuit = f.compile(inputset)

sample = [
    [0, 1],
    [2, 3],
    [3, 0],
]
expected_output = [
    [0, 1],
    [4, 27],
    [9, 0]
]
actual_output = circuit.encrypt_run_decrypt(np.array(sample))

for i in range(3):
    for j in range(2):
        if j == 0:
            assert actual_output[i][j] == expected_output[i][j] == squared[sample[i][j]]
        else:
            assert actual_output[i][j] == expected_output[i][j] == cubed[sample[i][j]]
```

In this example, we applied a `squared` table to the first column and a `cubed` table to the second column.

## Fused table lookup

**Concrete** can automatically fuse some operations into table lookups so that you don't have to create lookup tables manually:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return (42 * np.sin(x)).astype(np.int64) // 10

inputset = range(8)
circuit = f.compile(inputset)

for x in range(8):
    assert circuit.encrypt_run_decrypt(x) == f(x)
```

{% hint style="info" %}
All lookup tables must be from integers to integers. Without `.astype(np.int64)`, **Concrete** will not be able to fuse.
{% endhint %}

The function is first traced into:

![](../\_static/tutorials/table-lookup/1.initial.graph.png)

**Concrete** then fuses appropriate nodes:

![](../\_static/tutorials/table-lookup/3.final.graph.png)

{% hint style="info" %}
Fusing makes the code more readable and easier to modify, so try to use it instead of manual `LookupTable`s whenever possible.
{% endhint %}

## Using automatically created table lookup

For explanations about the `fhe.univariate(function)` and `fhe.multivariate(function)` features, which are convenient ways to use automatically created table lookups, refer to [this page](extensions.md).

## Table lookup exactness

TLUs are performed with an FHE operation called `Programmable Bootstrapping` (PBS). PBSs have a certain probability of error, which can result in inaccurate results.

Consider the following table:

```python
lut = [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

If you perform a Table Lookup using `4`, the expected result is `lut[4] = 16`. However, due to the possibility of error, you could receive any other value in the table.

The probability of this error can be configured using the `p_error` and `global_p_error` configuration options:
- `p_error` applies to individual TLUs
- `global_p_error` applies to the whole circuit

For example, if you set `p_error` to `0.01`, each TLU in the circuit will have a 99% (or greater) chance of being exact. With only one TLU in the circuit, it corresponds to `global_p_error = 0.01`. However, with two TLUs, `global_p_error` would be higher: `1 - (0.99 * 0.99) ≈ 0.02 = 2%`.

Setting `global_p_error` to `0.01` ensures that the entire circuit will have at most a `1%` probability of error, regardless of the number of TLUs. In this case, `p_error` will be smaller than `0.01` if there is more than one TLU.

If both `p_error` and `global_p_error` are set, the stricter condition will apply.

By default, both `p_error` and `global_p_error` are set to `None`, resulting in a `global_p_error` of `1 / 100_000` as default.

You can adjust these configuration options to suit your needs. See [How to Configure](../guides/configure.md) to learn how you can set a custom `p_error` and/or `global_p_error`.

{% hint style="info" %}
Configuring these variables have significant impact: 
- **Compilation and execution times**: compilation, key generation, circuit execution
- **Space requirements**: key sizes on disk and in memory.

In general, lower error probabilities result in longer compilation and execution times and larger space requirements.
{% endhint %}

## Table lookup performance

PBSs are computationally expensive. To optimize the performance, in some cases you can replace PBS by [rounded PBS](rounding.md), [truncate PBS](truncating.md) or [approximate PBS](rounding.md). With lightly different semantics, these TLUs offer more efficiency without sacrificing accuracy, which can very useful in cases like machine learning.