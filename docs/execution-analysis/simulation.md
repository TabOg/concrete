# Simulation

During development, the speed of homomorphic execution can be a blocker for fast prototyping. You could call the function you're trying to compile directly, of course, but it won't be exactly the same as FHE execution, which has a certain probability of error (see [Exactness](../core-features/table\_lookups.md#table-lookup-exactness)).

To overcome this issue, simulation is introduced:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return (x + 1) ** 2

inputset = [np.random.randint(0, 10, size=(10,)) for _ in range(10)]
circuit = f.compile(inputset, p_error=0.1, fhe_simulation=True)

sample = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

actual = f(sample)
simulation = circuit.simulate(sample)

print(actual.tolist())
print(simulation.tolist())
```

After the simulation runs, it prints the following:

```
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
[1, 4, 9, 16, 16, 36, 49, 64, 81, 100]
```

{% hint style="warning" %}
There are some operations which are not supported in simulation yet. They will result in compilation failures. You can revert to simulation using graph execution using `circuit.graph(...)` instead of `circuit.simulate(...)`, which won't simulate FHE, but it will evaluate the computation graph, which is like simulating the operations without any errors due to FHE.
{% endhint %}

## Overflow Detection in Simulation

Overflow can happen during an FHE computation, leading sometimes to weird behaviors. Using simulation can help you detect such event. It prints a warning whenever an overflow happen.

We can use the above circuit to trigger some overflow:

```python
>>> circuit.simulate([0,1,2,3,4,5,6,7,8,15])
WARNING at loc("<stdin>":3:0): overflow happened during addition in simulation
array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 0])
```

{% hint style="info" %}
Here I got `loc("<stdin>":3:0)` because I'm running from the interpreter. A filename will be used instead if you are running a Python script (e.g. `python filename.py`).
{% endhint %}

If you look at the MLIR (`circuit.mlir`), you will see that the input is supposed to be of type `eint4` thus represented in 4 bits (15 maximum). Since there's an addition of the input, we used the maximum value (15) here to trigger an overflow (15 + 1 = 16 which needs 5 bits). The warning tells us about the operation that did the overflow, as well as its location. You will get similar warnings for all basic FHE operations (add, mul, and lookup tables).
