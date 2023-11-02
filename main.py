import math
import random
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from os import path

# A class that will help to track trace of operations performed on values
# FIX: calling ._backward on a node more than once overwrites instead of accumulating, replaced = for +=
# IMPLEMENTATION: allow operations between Value class objects and numbers
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
        other = other if isinstance(other, Value) else Value(other)
        # out = parent node, self and other = children nodes
        out = Value(self.data + other.data, (self, other), "+")
        # calculating local gradients by backpropagation (including chain rule)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out


    def __radd__(self, other):
        return self + other


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # out = parent node, self and other = children nodes
        out = Value(self.data * other.data, (self, other), "*")
        # calculating local gradients by backpropagation (including chain rule)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out


    def __rmul__(self,other):
        return self * other


    def exp(self):
        x = self.data
        out = Value(data=math.exp(x), _children=(self, ), _op="e")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out


    def __pow__(self, other):
        assert isinstance(other, (int,float))
        out = Value(data=self.data**other, _children=(self, ), _op=f"**{other}")
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    

    def __neg__(self):
        return self * -1
    
    def __rsub__(self, other):
        return self - other
    

    def __sub__(self, other):
        return self + (-other)


    def __truediv__(self, other):
        return self * other**-1


    def tanh(self):
        x = self.data
        tanh = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(data=tanh, _children=(self, ), _op="tanh")
        def _backward():
            self.grad += (1 - tanh**2) * out.grad
        out._backward = _backward
        return out


    def backward(self):
        topo = []
        visited = set()
        def topological_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topological_sort(child)
                topo.append(node)
        topological_sort(self)
        topo.reverse()
        self.grad = 1.00
        for node in topo:
            node._backward()
        return topo



from graphviz import Digraph


def draw(root):
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
    nodes, edges = trace(root)
    graph = Digraph(name="Equation relationship", format="png")
    for node in nodes:
        graph.node(name=str(id(node)), label=f"{node.label} | data = {node.data} | grad = {node.grad}", shape="record") 
        if node._op:
            graph.node(name=f"{id(node)}{node._op}", label=node._op)
            graph.edge(f"{id(node)}{node._op}", str(id(node)))
    for tail, head in edges:
        graph.edge(str(id(tail)), f"{id(head)}{head._op}")
    graph.render(directory=path.dirname(__file__), view=True)


#IMPLEMENTATION: Topological sort
#More than a topological sort this seems more like a DFS algorithm
#root->child1->child1_child1
def topological_sort(node: Value, visited:set, topo:[]):
    if node not in visited:
        visited.add(node)
        for child in node._prev:
            topological_sort(child, visited, topo)
        topo.append(node)


class Neuron:
    def __init__(self, nin):
        #create a random weight for every of the inputs provided in number of inputs (nin)
        self.w = [Value(random.uniform(1, -1)) for _ in range(nin)]
        #generate a random bias for the neuron
        self.b = Value(random.uniform(1, -1))
    
    #this methods will execute when you call an instance of the neuron call with () as suffix
    def __call__(self, x):
        #w * x + b 
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
        #This should print the same as the list comprehension version
        # total = 0
        # for i in range(len(self.w)):
        #     total += x[i] * self.w[i]
        # total += self.b
        # print(total)
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n = MLP(3, [4,4,1])
xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
ys = [1.0, -1.0, -1.0, 1.0]#desired outputs
# n(x)
ypred = [n(x) for x in xs]
print(ypred)
print(n.parameters())  
for ygt, yout in zip(ys, ypred):
    print(ygt, yout)
loss = sum((yd - ya)**2 for yd, ya in zip(ys, ypred))/2*(len(xs))
print(loss)
#reducing loss with gradient descent
for i in range(200):
    #forward pass
    ypred = [n(x) for x in xs]
    #calculating the mean squared error loss function
    loss = sum((y_approximation - y_desired)**2 for y_desired, y_approximation in zip(ys, ypred))
    #after calculating the loss we need to reset the gradients
    for p in n.parameters():
        p.grad = 0.0
    #backward pass 
    loss.backward()
    #calculating the new weights and biases with gradient descent w = w-step(dJ(w,b)/dw) and b = b-step(dJ(w,b)/db)
    for p in n.parameters():
        p.data += -0.01 * p.grad
    print(i, loss.data)
print(ypred)
