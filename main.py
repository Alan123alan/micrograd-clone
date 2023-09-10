import math
import numpy as np
import matplotlib.pyplot as plt
from os import path

# A basic function of x to ponder on what a derivative is
def f(x):
    return 3*x**2 - 4*x + 5

# Evaluating a f(x) will return a value (y) that we could use together with x to represent a point in a plane
x = 3
y = f(x)
print(y)

# Understanding numerical derivative (the slope)
# In mathematics, the derivative shows the sensitivity of change of a function's output with respect to the input.
# Limit of a function of x as it approaches h = 0
h = 0.000001
slope = (f(x + h) - f(x))/h
print(slope)
# The slope shows how much the result of the function responds to a change in the input
# At some point the slope could be 0 meaning that a change in the input causes no change in the output
x = 2/3
slope = (f(x + h) - f(x))/h
print(slope)

# A basic function that depends on 3 inputs a, b and c
def f(a,b,c):
        return a*b+c
a = 2
b = -3
c = 10
d1 = f(a,b,c)
# If a is increased by a bit
# Meaning that the function is derived with respect to a, a->1 so the result of derived function (slope) only depends on b and c
# a += h
# If b is increased by a bit
# Meaning that the function is derived with respect to b, b->1 so the result of derived function (slope) only depends on a and c
b += h
d2 = f(a,b,c)
print(f"D1 : {d1}")
print(f"D2 : {d2}")
print(f"Slope : {(d2-d1)/h}")

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out
        
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(a)
print(b)
print(c)
print(d._prev)

from graphviz import Digraph
graph = Digraph(name="Equation relationship", format="png")
graph.node(name=str(id(a)), label=f"Data = {a.data})", shape="record")
graph.node(name=str(id(b)), label=f"Data = {b.data})", shape="record")
graph.edge(str(id(a)), str(id(b)))
print(graph.source)
print(id(graph))
graph.render(directory=path.dirname(__file__), view=True)