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
To know how the output of the function will respond if any of it's inputs changes by a bit, we can bump all of the inputs and see how   
the output changes.   
  
If the `a` input is increased by a small amount `h` this means that the function should be derived with respect to `a`,  
and since `d(a)/da->1` the result of derived function (slope) only depends on `b` and `c`.

```Python
a = 2 + h
b = -3
c = 10
d2 = f(a,b,c)
```
If the `b` input is increased by a small amount `h` this means that the function should be derived with respect to `b`,  
and since `d(b)/db->1` the result of derived function (slope) only depends on `a` and `c`.

```Python
a = 2
b = -3 + h
c = 10
d2 = f(a,b,c)
```

If the `c` input is increased by a small amount `h` this means that the function should be derived with respect to `c`,  
and since `d(c)/dc->1` the result of derived function (slope) only depends on `a` and `b`.

```Python
a = 2
b = -3
c = 10 + h
d2 = f(a,b,c)
```
# Instantiating value objects 
a = Value(2.0,label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
d = a * b
e = d + c
d.label = "d"
e.label = "e"
e.grad = 1.00
d.grad = 1.00
c.grad = 1.00
a.grad = -3.00
b.grad = 2.00
# Checking the objects are set correctly
# print(f"a data: {a}, children: {a._prev}, op: {a._op}")
# print(f"b data: {b}, children: {b._prev}, op: {b._op}")
# print(f"c data: {c}, children: {c._prev}, op: {c._op}")
# print(f"d data: {d}, children: {d._prev}, op: {d._op}")


def manual_backpropagation():
    h = 0.00001
    a = Value(2.0,label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    d = a * b
    e1 = d + c
    d.label = "d"
    e1.label = "e"
    b.data += h
    d = a * b
    # d.data += h
    e2 = d + c
    slope = (e2.data-e1.data)/h
    # print(slope)
    # To current node (with exception of the head node which grad is the 1 because dhead/dhead = 1)
    # So if a*b = d and c + d = e
    # de/de = 1 so this will be the grad for the head node
    # Then we need to derive de/dd and de/dc to get the grads (slopes)
    # de/dd with e = d + c -> d(d+c)/dd
    # By applying the definition of derivative f(x+h)-f(x)/h
    # We have ((d+h+c)-(d+c))/h -> 1.00
    # And for de/dc ((d+c+h)-(d+c))/h -> 1.00
    # Now to get how does a and b affect e it gets a bit more complicated
    # Since a and b only affect e through their influence on d
    # And this matches with the definition of the chain rule from calculus
    # If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are dependent variables), then z depends on x as well, via the intermediate variable y.
    # This can be expressed as dz/dx = dz/dy * dy/dx
    # So if e depends on d which depends also on a we have
    # de/da = de/dd * dd/da -> de/da = 1.00 * d(a * b)/da
    # (a+h * b)-(a * b)/h -> a*b + h*b - a*b/h -> h*b/h -> b
    # Then de/da = 1.00 * b, for b = -3.00 de/da = -3.00
    # de/db = de/dd * dd/db -> de/db = 1.00 * d(a * b)/db
    # (a * b+h)-(a * b)/h -> a*b + h*a - a*b/h -> h*a/h -> a
    # Then de/db = 1.00 * a, for a = 2.00 de/db = 2.00
    # An interesting note is that plus nodes take the parent gradient and distribute it to the children nodes
manual_backpropagation()

def neuron_implementation():
    x1 = Value(data=2.0, label="x1")
    x2 = Value(data=0.0, label="x2")
    w1 = Value(data=-3.0, label="w1")
    w2 = Value(data=1.0, label="w2")
    b = Value(data=6.7, label="b")
    x1w1 = x1*w1; x1w1.label = "x1w1"
    x2w2 = x2*w2; x2w2.label = "x2w2"
    x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label = "x1w1+x2w2"
    neuron_cell_body = x1w1_x2w2 + b; neuron_cell_body.label = "neuron cell body"
    output = neuron_cell_body.tanh(); output.label = "output"
    # nodes, edges = trace(output)
    # Output after activation function
    # draw(nodes, edges)

neuron_implementation()

def neuron_backpropagation():
    x1 = Value(data=2.0, label="x1")
    x2 = Value(data=0.0, label="x2")
    w1 = Value(data=-3.0, label="w1")
    w2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")
    x1w1 = x1*w1; x1w1.label = "x1w1"
    x2w2 = x2*w2; x2w2.label = "x2w2"
    x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label = "x1w1+x2w2"
    n = x1w1_x2w2 + b; n.label = "neuron cell body"
    o = n.tanh(); o.label = "output"
    # do/do = 1 so the gradient at the head is 1.0
    o.grad = 1.00
    # Let's start backpropagation, then next is do/dn and o = tanh(n)
    # Applying the derivative definition  f(x+h) - f(x)/h when h -> 0
    # tanh(n+h) - tan(n)/h this seems confusing, in the video it is recommended to use d tanh(x)/dx = 1 - tanh(x)**2
    # By using that derivative and knowing that o = tanh(n), then d tanh(n)/dn = 1 - o**2
    n.grad = 1.00 - (o.data**2)
    # Let's continue backpropagating to b now, we are looking for do/db, by the chain rule we know that
    # do/db = do/dn * dn/db, and n = x1w1x2w2 + b
    # (x1w1x2w2 + b + h) - (x1w1x2w2 + b)/h -> h/h -> 1.00
    # do/dn = 1.00 - (o.data**2) and dn/db = 1.00 so do/db = 1.00 * (1.00 -(o.data**2))
    # b.grad = 1.00 * (1.00 - (o.data**2)) # worked basically for no reason since it was previously stated that '+' nodes only pass gradients
    b.grad = n.grad
    x1w1_x2w2.grad = n.grad
    # Since do/dx1w1 and do/dx2w2 will both be 1.00 because they are added nodes
    # Then applying the chain rule will get that x1w1 grad = 1.0 * (x1w1 + x2w2 grad) 
    # Then applying the chain rule will get that x2w2 grad = 1.0 * (x1w1 + x2w2 grad) 
    x1w1.grad = x1w1_x2w2.grad
    x2w2.grad = x1w1_x2w2.grad
    # Now we are at the start and we need to get do/x2 and do/w2
    # First do/x2 = do/dx2w2 * dx2w2/x2
    # dx2w2/x2 -> (x2+h * w2) - (x2 * w2)/h -> (x2 * w2 + h * w2 - x2 * w2)/h -> (h * w2)/h -> w2
    # And we know that we must multiply w2 times the previous grad to get do/dx2
    x2.grad = w2.data * x2w2.grad
    # By the same proces dx2w2/dw2 -> d(x2 * w2)/dw2 -> (x2 * (w2+h) - (x2 * w2))/h -> (x2*h + x2*w2 - x2w2)/h -> x2
    w2.grad = x2.data * x2w2.grad
    # By logic
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad
    # nodes, edges = trace(o)
    # draw(nodes, edges)
neuron_backpropagation()

#MODIFICATION: Implementing automatic backpropagation
#NOTE: Still need to manually set the output gradient
def neuron_automatic_backpropagation():
    x1 = Value(data=2.0, label="x1")
    x2 = Value(data=0.0, label="x2")
    w1 = Value(data=-3.0, label="w1")
    w2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")
    x1w1 = x1*w1; x1w1.label = "x1w1"
    x2w2 = x2*w2; x2w2.label = "x2w2"
    x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label = "x1w1+x2w2"
    n = x1w1_x2w2 + b; n.label = "neuron cell body"
    # o = n.tanh(); o.label = "output"
    e = (2*n).exp()
    o = (e - 1) / (e + 1); o.label = "output"
    # Since the tanh result value backpropagation function applies the chain rule by default
    # we need to manually set o.grad for backpropagation to work as expected
    # o.grad = 1.00
    o.backward()
    # visited = set()
    # topo = []
    # topological_sort(o, visited, topo)
    # topo.reverse()
    # for node in topo:
        # node._backward()
    # o._backward()
    # n._backward()
    # x1w1_x2w2._backward()
    # x2w2._backward()
    # x1w1._backward()
    # The below is not needed for x1, w1, x2 and w2 because the previous calls to ._backward set those gradients
    # x2._backward()
    # w2._backward()
    # x1._backward()
    # w1._backward()
    # nodes, edges = trace(o)
    # draw(nodes, edges)
neuron_automatic_backpropagation()
# TO DO: Do I need to include a step or conditional to not apply chain rule for multiplication result nodes?
# TO DO: implement _backward for exp method in Value class 
a = Value(data=32)
b = 1 + a
c = 3 * b
# The previous expression is equivalent to 1.__add__(a)
# Let's check
type(a)
type(1)
print(b)
print(c)