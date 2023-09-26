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
        self._backward = lambda:None
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        # out = parent node, self and other = children nodes
        out = Value(self.data + other.data, (self, other), "+")
        # calculating local gradients by backpropagation (including chain rule)
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        # out = parent node, self and other = children nodes
        out = Value(self.data * other.data, (self, other), "*")
        # calculating local gradients by backpropagation (including chain rule)
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        tanh = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(data=tanh, _children=(self, ), _op="tanh")
        def _backward():
            self.grad = (1 - tanh**2) * out.grad
        out._backward = _backward
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
a.grad = -3.00
b.grad = 2.00
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


def draw(nodes, edges):
    graph = Digraph(name="Equation relationship", format="png")
    for node in nodes:
        graph.node(name=str(id(node)), label=f"{node.label} | data = {node.data} | grad = {node.grad}", shape="record") 
        if node._op:
            graph.node(name=f"{id(node)}{node._op}", label=node._op)
            graph.edge(f"{id(node)}{node._op}", str(id(node)))
    for tail, head in edges:
        graph.edge(str(id(tail)), f"{id(head)}{head._op}")
    # print(graph.source)
    graph.render(directory=path.dirname(__file__), view=True)

# nodes, edges = trace(e)
# draw(nodes, edges)
# print(nodes, edges)

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
    print(slope)
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
    # An interesting note is that plus nodes take the parent gradient and distribute it tho the children nodes
    
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
    o = n.tanh(); o.label = "output"
    # Since the tanh result value backpropagation function applies the chain rule by default
    # we need to manually set o.grad for backpropagation to work as expected
    o.grad = 1.00
    o._backward()
    n._backward()
    x1w1_x2w2._backward()
    x2w2._backward()
    x1w1._backward()
    # The below is not needed for x1, w1, x2 and w2 because the previous calls to ._backward set those gradients
    # x2._backward()
    # w2._backward()
    # x1._backward()
    # w1._backward()
    nodes, edges = trace(o)
    draw(nodes, edges)
neuron_automatic_backpropagation()
# TO DO: Do I need to include a step or conditional to not apply chain rule for multiplication result nodes?
# TO DO: Implement topological sort on the nodes to be able to implement backward progapation for a whole expression
stack = []
visited = set()
#More than a topological sort this seems more like a DFS algorithm
#root->child1->child1_child1
def topological_sort(node: Value):
    if node not in visited:
        visited.add(node)
        for child in node._prev:
            topological_sort(child)
        stack.append(node)
d = Value(data=2)
e = Value(data=3)
c = d + e
b = Value(data=4)
a = b + c
topological_sort(a)
print(stack)