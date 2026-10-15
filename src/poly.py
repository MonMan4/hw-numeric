'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree.

>>> p = Poly([1, 2, 3])
>>> print(p)
3x**2 + 2x + 1

Polynomials are just functions with special structure,
and they can be called like regular functions.

>>> p(3)
34

Because of this special structure, we can do a lot of operations that can't be done on ordinary functions.

We can compute derivatives:

>>> print(p.derivative())
6x + 2

We can check for equality:

>>> q = Poly([0, 3, 2, 1])
>>> print(q)
x**3 + 2x**2 + 3x
>>> p == q
False

We can do addition, subtraction, multiplication, and scalar exponentiation:

>>> print (p + q)
x**3 + 5x**2 + 5x + 1
>>> print (p - q)
-x**3 + x**2 - x + 1
>>> print (p * q)
3x**5 + 8x**4 + 14x**3 + 8x**2 + 3x
>>> print(p ** 2)
9x**4 + 12x**3 + 10x**2 + 4x + 1

Technically, polynomials form a *Ring*
(a type of structure studied in MATH171 Abstract Algebra),
and so they do not support division.
(If they did, they would be called a *Field*.)
'''


class Poly:
    '''
    A dense polynomial with rational coefficients.

    The field .coeffs is a list of length at least 1, from lowest to
    highest degree, with no trailing zeros.  Coeffs[-1] is the leading
    coefficient.

    >>> p = Poly([1, 2, 3])
    >>> p.coeffs
    [1, 2, 3]
    >>> Poly([1, 2, 0, 0]).coeffs
    [1, 2]
    >>> Poly([]).coeffs
    [0]

    The `==` operator compares polynomials by value, and polynomials
    that represent the same function are equal regardless of how their
    coefficient lists were passed in:

    >>> Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    True
    >>> Poly([1, 2]) == 1
    False

    `Poly` is a mutable value type, so it is deliberately unhashable
    and cannot be used as a dict key or a set element:

    >>> hash(Poly([1, 2, 3]))
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'Poly'

    The `repr` is the constructor call, so `eval(repr(p)) == p` for
    every Poly.  The `str` is the usual algebraic notation, with terms
    in descending degree, unit coefficients suppressed, and zero
    coefficients skipped.  The zero polynomial prints as `0`:

    >>> eval(repr(Poly([1, 2, 3]))) == Poly([1, 2, 3])
    True
    >>> print(Poly([1, 2, 3]))
    3x**2 + 2x + 1
    >>> print(Poly([0]))
    0
    >>> print(Poly([5]))
    5
    >>> print(Poly([-5]))
    -5
    >>> print(Poly([1, 1]))
    x + 1
    >>> print(Poly([0, 1]))
    x
    >>> print(Poly([0, 0, -1]))
    -x**2
    >>> print(Poly([1, -1, 0, 1]))
    x**3 - x + 1
    >>> print(Poly([1, 0, 2]))
    2x**2 + 1
    '''
    def __init__(self, coeffs):
        # Normalize: strip trailing zeros, guarantee at least [0].
        c = list(coeffs)
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        if not c:
            c = [0]
        self.coeffs = c

    def __repr__(self):
        return f'Poly({self.coeffs!r})'

    def __str__(self):
        terms = []
        for i in range(len(self.coeffs) - 1, -1, -1):
            c = self.coeffs[i]
            if c == 0:
                continue
            # magnitude string for the term (without its sign)
            if i == 0:
                body = f'{abs(c)}'
            else:
                var = 'x' if i == 1 else f'x**{i}'
                if abs(c) == 1:
                    body = var
                else:
                    body = f'{abs(c)}{var}'
            if not terms:
                # first (highest-degree) term decides overall sign
                terms.append(('-' if c < 0 else '') + body)
            else:
                terms.append((' - ' if c < 0 else ' + ') + body)
        if not terms:
            return '0'
        return ''.join(terms)

    def __eq__(self, other):
        if isinstance(other, Poly):
            return self.coeffs == other.coeffs
        return NotImplemented

    # __hash__ must be explicitly disabled: defining __eq__ sets it to
    # None automatically, but being explicit documents the intent.
    __hash__ = None

    def __call__(self, x):
        return self.eval_horner(x)

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __add__(self, other):
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        out = [0] * n
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            out[i] = ai + bi
        return Poly(out)

    def __sub__(self, other):
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        out = [0] * n
        for i in range(n):
            ai = a[i] if i < len(a) else 0
            bi = b[i] if i < len(b) else 0
            out[i] = ai - bi
        return Poly(out)

    def __mul__(self, other):
        return self.mul_naive(other)

    def derivative(self):
        if len(self.coeffs) <= 1:
            return Poly([0])
        return Poly([i * c for i, c in enumerate(self.coeffs)][1:])

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError('Poly exponent must be a non-negative int')
        result = Poly([1])
        for _ in range(n):
            result = result * self
        return result

    # in-place ops: mutate self.coeffs, return self (tests check id)
    def __iadd__(self, other):
        self.coeffs = (self + other).coeffs
        return self

    def __isub__(self, other):
        self.coeffs = (self - other).coeffs
        return self

    def __imul__(self, other):
        self.coeffs = (self * other).coeffs
        return self

    def __ipow__(self, n):
        self.coeffs = (self ** n).coeffs
        return self

    def mul_karatsuba(self, other):
        return Poly(_karatsuba(self.coeffs, other.coeffs))

    def eval_naive(self, x):
        '''
        Evaluate by summing a_i * x**i one term at a time.  The
        per-term `x ** i` recomputes a power each iteration, so the
        runtime is Theta(n) power lifts, each Theta(i) for the
        exponentiation, giving Theta(n**2) overall.

        >>> Poly([1, 2, 3]).eval_naive(2)
        17
        '''
        total = 0
        for i, c in enumerate(self.coeffs):
            total += c * x ** i
        return total

    def eval_horner(self, x):
        '''
        Evaluate with Horner's rule:

            (((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

        This is Theta(n) and uses only one multiply and one add per
        coefficient.  It is also numerically more stable than the
        naive evaluation when consecutive coefficients have opposite
        signs, because no large intermediate x**i is computed.

        >>> Poly([1, 2, 3]).eval_horner(2)
        17
        '''
        total = 0
        for c in reversed(self.coeffs):
            total = total * x + c
        return total

    def mul_naive(self, other):
        '''
        Naive multiplication uses Theta(n**2) multiplications.

        >>> Poly([1, 2]) .mul_naive(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        a, b = self.coeffs, other.coeffs
        if not a or not b:
            return Poly([0])
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return Poly(out)


def _karatsuba(a, b):
    '''
    Product of two coefficient lists by Karatsuba's algorithm.

    The inputs are coefficient lists, lowest degree first.
    The return value is the coefficient list of the product.
    The recursion splits both operands at the midpoint and
    uses only three half-size products.
    That gives the recurrence

        T(n) = 3 T(n/2) + Theta(n)
             = Theta(n^log2 3)          # by master theorem
             = O(n^1.586)               # by defn of O()

    See <https://www.geeksforgeeks.org/dsa/multiply-two-polynomials-2/> for a basic tutorial/implementation you can adapt for your code.

    >>> _karatsuba([1, 2], [3, 4])
    [3, 10, 8]
    >>> _karatsuba([1], [1])
    [1]
    >>> _karatsuba([0], [1, 2, 3])
    [0, 0, 0]
    '''
    # Base case: small inputs use schoolbook multiplication.  Without
    # this cutoff the recursion never bottoms out and the slice
    # overhead swamps the asymptotic gain.
    if not a or not b:
        return [0]
    if len(a) <= 2 or len(b) <= 2:
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return out

    # Split at the midpoint of the longer operand so both halves are
    # non-empty and the recursion is balanced.
    n = max(len(a), len(b))
    m = n // 2
    a0, a1 = a[:m], a[m:]
    b0, b1 = b[:m], b[m:]

    z0 = _karatsuba(a0, b0)
    z2 = _karatsuba(a1, b1)

    # (a0 + a1)(b0 + b1) = z0 + z1 + z2  =>  z1 = middle - z0 - z2
    a_sum = _add_lists(a0, a1)
    b_sum = _add_lists(b0, b1)
    z1 = _sub_lists(_sub_lists(_karatsuba(a_sum, b_sum), z0), z2)

    # Reassemble: z0 + z1 * x^m + z2 * x^(2m)
    out = [0] * (len(a) + len(b) - 1)
    for i, c in enumerate(z0):
        out[i] += c
    for i, c in enumerate(z1):
        out[i + m] += c
    for i, c in enumerate(z2):
        out[i + 2 * m] += c
    return out


def _add_lists(a, b):
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
    return out


def _sub_lists(a, b):
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)
    return out
