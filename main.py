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

# A class that will help to track trace of operations performed on values
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

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
# Checking the objects are set correctly
print(f"a data: {a}, children: {a._prev}, op: {a._op}")
print(f"b data: {b}, children: {b._prev}, op: {b._op}")
print(f"c data: {c}, children: {c._prev}, op: {c._op}")
print(f"d data: {d}, children: {d._prev}, op: {d._op}")

from graphviz import Digraph
def trace(root):
    nodes, edges = set(), set()
    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                build(child)
    build(root)
    return nodes, edges
nodes, edges = trace(e)
print(nodes, edges)
graph = Digraph(name="Equation relationship", format="png")
for node in nodes:
    graph.node(name=str(id(node)), label=f"{node.label} | data = {node.data} | grad = {node.grad}", shape="record") 
    if node._op:
        graph.node(name=f"{id(node)}{node._op}", label=node._op)
        graph.edge(f"{id(node)}{node._op}", str(id(node)))
for tail, head in edges:
    graph.edge(str(id(tail)), f"{id(head)}{head._op}")
print(graph.source)
graph.render(directory=path.dirname(__file__), view=True)


def manual_backpropagation():
    h = 0.00001
    a = Value(2.0,label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    d = a * b
    e1 = d + c
    d.label = "d"
    e1.label = "e"
    d = a * b
    d.data += h
    e2 = d + c
    slope = (e2.data-e1.data)/h
    print(slope)
    # The grad of a current node
    # Is the derivate of previous node with respect
    # To current node (with exception of the head node)
    # So if a*b = d and c + d = e
    # de/de = 1 so this will be the grad for the head node
    # Then we need to derive de/dd and de/dc to get the grads (slopes)
    # de/dd with e = d + c -> d(d+c)/dd
    # By applying the definition of derivative f(x+h)-f(x)/h
    # We have ((d+h+c)-(d+c))/h -> 1.00
    # And for de/dc ((d+c+h)-(d+c))/h -> 1.00
manual_backpropagation()

