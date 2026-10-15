'''
This file implements automatic differentiation (autodiff) with dual numbers.
Autodiff is the backbone of modern machine learning libraries like torch and tensorflow because these libraries train models using the gradient descent algorithm.

First, let's see an example.
Consider the function below.

>>> f = lambda x: sin(x ** 2) + 2 * x

With some basic calculus, we can hand-compute the derivative as

>>> df = lambda x: 2 * x * cos(x ** 2) + 2

But hand computing derivatives is error prone.
And some derivatives do not have a closed form solution.
For example, the *recursive neural network* (RNN) is a popular model.
The function below shows a simple example RNN.

>>> def rnn(xs, w, h=0.0):
...     if not xs:
...         return h
...     return rnn(xs[1:], w, tanh(w * h + xs[0]))

Observe that the function takes a list of numbers as the input and recursively calls itself on the list.
Computing the derivative of this function by hand is a nightmare.

The magic of autodiff is that dual numbers let us compute the derivative automatically for *all* functions f.
And in practice, it is common to compute the derivative of functions  much more complicated than the rnn function above.

The procedure is simple but counter-intuitive:

1. Define a Dual object whose v is the location where you want to take the derivative, and d=1.

    >>> x = Dual(v=0, d=1)

2. Call the function on our Dual variable.
    >>> f(x)
    Dual(v=0.0, d=2.0)

    This returns another Dual, and getting the .d attribute gives us the derivative.
    >>> f(x).d
    2.0

3. We can confirm that this matches the hand computed derivative:
    >>> df(0)
    2.0

Now we can compute the derivative of the rnn function as:

>>> xs = [1.0, 1.0]         # this is our "dataset" we are learning
>>> w = Dual(v=0.5, d=1)    # this is the initialization procedure
>>> h = rnn(xs, w)          # h = "hidden" is the trad rnn result
>>> round(h.d, 4)
0.1703

> **NOTE**
> The doctest above uses round() for numeric stability reasons.
> Because your implementation of the Dual number is likely to be subtly different than mine,
> the last few digits of the number will be slightly different,
> even if your implementation is mathematically correct.
> Rounding floats in tests is common practice.

So how does this magic work?

A dual number is a pair (v, d) where v is the value and d the derivative.
When defined our variable to be

>>> x = Dual(v=3, d=1)

The reason we started with d=1 is because the derivative of x is 1.
Mathematicians often represent this "derivative" as an infinitesimal number ε, and so the __str__ output of Dual looks like

>>> print(x)
3 + 1ε

When performing any mathematical computations,
we do the normal math operations on the v values,
and we apply the chain rule to generate the d values:

>>> x = Dual(v=3, d=1)
>>> y = Dual(v=2, d=1)
>>> print(x + y)
5 + 2ε
>>> print(x - y)
1 + 0ε
>>> print(x * y)
6 + 5ε
>>> print(x ** 2)
9 + 6ε

Observe that unlike the Poly class, Dual numbers form a field.
And so division is well-defined:

>>> print(x / y)
1.5 + -0.25ε

Python has dunder methods for all the standard math operators,
but does not have dunder methods for the transcendental functions like sin, cos, exp, and log.
The module has redefined these functions so that they work on both Dual numbers and standard floats.

>>> sin(Dual(0.0, 1.0))
Dual(v=0.0, d=1.0)
>>> exp(Dual(0.0, 1.0))
Dual(v=1.0, d=1.0)
'''
import math


class Dual:
    '''
    The dual number v + d*epsilon with epsilon**2 == 0.

    The fields are .v (the value) and .d (the derivative with respect
    to whatever variable the caller has seeded with derivative 1).

    Every arithmetic method returns a new Dual and never mutates self.

    >>> Dual(1, 2) == Dual(1, 2)
    True
    >>> Dual(1, 2) == Dual(1, 3)
    False
    >>> Dual(1, 2) == 1
    False
    >>> hash(Dual(1, 2)) == hash(Dual(1, 2))
    True

    The `repr` is the constructor call, so `eval(repr(x))` recovers x,
    and `str` is the usual `v + d*epsilon` notation:

    >>> eval(repr(Dual(1, 2))) == Dual(1, 2)
    True
    >>> print(Dual(1, 2))
    1 + 2ε
    >>> print(Dual(1, 2) + Dual(3, 4))
    4 + 6ε
    '''
    def __init__(self, v, d):
        self.v = v
        self.d = d

    def __repr__(self):
        return f'Dual(v={self.v!r}, d={self.d!r})'

    def __str__(self):
        return f'{self.v} + {self.d}ε'

    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.v == other.v and self.d == other.d
        return NotImplemented

    def __hash__(self):
        return hash((self.v, self.d))

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v + other.v, self.d + other.d)
        return Dual(self.v + other, self.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v - other.v, self.d - other.d)
        return Dual(self.v - other, self.d)

    def __rsub__(self, other):
        return Dual(other - self.v, -self.d)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v * other.v, self.d * other.v + self.v * other.d)
        return Dual(self.v * other, self.d * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v / other.v,
                        (self.d * other.v - self.v * other.d) / (other.v ** 2))
        return Dual(self.v / other, self.d / other)

    def __rtruediv__(self, other):
        return Dual(other / self.v, -other * self.d / (self.v ** 2))

    def __pow__(self, other):
        if isinstance(other, Dual):
            v_val = self.v ** other.v
            d_val = v_val * (other.d * math.log(self.v) + other.v * self.d / self.v)
            return Dual(v_val, d_val)
        return Dual(self.v ** other, other * self.v ** (other - 1) * self.d)


def sin(x):
    '''
    sin, differentiable if x is a Dual.

    >>> sin(0.0)
    0.0
    '''
    if isinstance(x, Dual):
        return Dual(math.sin(x.v), math.cos(x.v) * x.d)
    return math.sin(x)


def cos(x):
    '''
    cos, differentiable if x is a Dual.

    >>> cos(0.0)
    1.0
    '''
    if isinstance(x, Dual):
        return Dual(math.cos(x.v), -math.sin(x.v) * x.d)
    return math.cos(x)


def tanh(x):
    '''
    tanh, differentiable if x is a Dual.

    >>> tanh(0.0)
    0.0
    '''
    if isinstance(x, Dual):
        t = math.tanh(x.v)
        return Dual(t, (1 - t ** 2) * x.d)
    return math.tanh(x)


def exp(x):
    '''
    exp, differentiable if x is a Dual.

    >>> exp(0.0)
    1.0
    '''
    if isinstance(x, Dual):
        e = math.exp(x.v)
        return Dual(e, e * x.d)
    return math.exp(x)


def log(x):
    '''
    log, differentiable if x is a Dual.

    >>> log(1.0)
    0.0
    '''
    if isinstance(x, Dual):
        return Dual(math.log(x.v), x.d / x.v)
    return math.log(x)
