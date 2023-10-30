# Micrograd clone 

This is a micrograd clone following Andrej Karpathy [video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

**Note** to use graphviz module you need to:
- pip install graphviz
- install graphviz in your computer


If you are using mac you need to `brew install graphviz`


A basic function of `x` to ponder on what a derivative is.
```Python
def f(x):
    return 3*x**2 - 4*x + 5
```

Evaluating a `f(x)` will return a value `y` that we could use together with `x` to represent a point in a plane.
```Python
x = 3
y = f(x)
print(y)
```

#### Understanding numerical derivative (the slope)

- Shows the sensitivity of change of a function's output with respect to the input.
- Limit of `f(x)` as `h -> 0`

```Python
h = 0.000001
slope = (f(x + h) - f(x))/h
print(slope)
```
The slope shows how much the result of the function responds to a change in the input,  
at some point the slope could be 0 meaning that a change in the input causes no change in the output.

```Python
x = 2/3
slope = (f(x + h) - f(x))/h
print(slope)
```

A function that depends on 3 inputs `a`, `b` and `c`.  

```Python
def f(a,b,c):
        return a*b+c
a = 2
b = -3
c = 10
d1 = f(a,b,c)
```
To know how the output of the function will respond if any of it's inputs changes by a bit, we can bump all of the inputs a see how the output changes.  
If the `a` input is increased by a small amount `h` this means that the function should be derived with respect to `a`, `d(a)/da->1` so the result of derived function (slope) only depends on `b` and `c`.
```Python
a = 2 + h
b = -3
c = 10
d2 = f(a,b,c)
```
# a += h
# If b is increased by a bit
# Meaning that the function is derived with respect to b, b->1 so the result of derived function (slope) only depends on a and c
b += h
d2 = f(a,b,c)
print(f"D1 : {d1}")
print(f"D2 : {d2}")
print(f"Slope : {(d2-d1)/h}")
