###############################################################################
# Problem 1....
###############################################################################
class Vector:
    """
    The 'Vector' class represents a vector of length 3.
    """
    ###########################################################################
    # Problem 1(a).
    ###########################################################################
    def __str__(self):
        """
        Return a string to print of this vector.
        """
        return "%r %r %r" % tuple(self.v)
    def __repr__(self):
        """
        Return the representation of this vector.
        """
        return "%s(%r, %r, %r)" % tuple([self.__class__.__name__] + self.v)
    ###########################################################################
    # Problem 1(b).
    ###########################################################################
    def __init__(self, v0, v1, v2):
        """
        Initialise the class with its four components.
        """
        self.v = [v0, v1, v2]
    ###########################################################################
    # Problem 1(c).
    ###########################################################################
    def __getitem__(self, i):
        """
        Return the 'i' component of the vector. Note, this can be a
        slice.
        """
        return self.v[i]
    def __setitem__(self, i, s):
        """
        Set the 'i' component of the vector with the scalar
        's'. Alternatively, 'i' can be a slice and 's' a sub-vector.
        """
        self.v[i] = s
    def __len__(self):
        """
        Return the length of the vector, e.g. 4.
        """
        return 3
    ###########################################################################
    # Problem 1(d).
    ###########################################################################
    def __pos__(self):
        """
        Return a copy of this vector with the '+' operator applied to
        each element.
        """
        return self.__class__(+self[0], +self[1], +self[2])
    def __neg__(self):
        """
        Return a copy of this vector with the '-' operator applied to
        each element.
        """
        return self.__class__(-self[0], -self[1], -self[2])
    def __iadd__(self, v):
        """
        Augmented assignment '+=' for adding a vector 'v' to this vector.
        """
        for i in range(0, 3): self[i] += v[i]
        return self
    def __isub__(self, v):
        """
        Augmented assignment '-=' for subtracting a vector 'v' from
        this vector.
        """
        for i in range(0, 3): self[i] -= v[i]
        return self
    def __add__(self, v):
        """
        Return the addition of this vector with the vector 'v'.
        """
        u = +self
        u += v
        return u
    def __sub__(self, v):
        """
        Return the subtraction of this vector with the vector 'v'.
        """
        u = +self
        u -= v
        return u
    ###########################################################################
    # Problem 1(e).
    ###########################################################################
    def __invert__(self):
        """
        Return the complex transpose of this vector.
        """
        v = +self
        for i in range(0, 3):
            try: v[i] = self[i].conjugate()
            except: v[i] = self[i]
        return v
    ###########################################################################
    # Problem 1(f), 1(g), and 2(g).
    ###########################################################################
    def __imul__(self, x):
        """
        Augmented assignment '*=' for multiplying this vector with a
        vector, matrix, or scalar 'x'.
        """
        # The vector case.
        try:
            x[0]
            self = sum(self[i]*x[i] for i in range(0, 4))
        except:
            # The matrix case.
            try:
                x[0, 0]
                u = +self
                for i in range(0, 3):
                    self[i] = sum([u[j]*x[j, i] for j in range(0, 4)])
            # The scalar case.
            except:
                for i in range(0, 3): self[i] *= x
        return self
    def __mul__(self, x):
        """
        Return the multiplication of this vector with a vector, matrix, or
        scalar 'x'.
        """
        u = +self
        u *= x
        return u
    def __rmul__(self, x):
        """
        Return the multiplication of a vector, matrix, or scalar 'x'
        with this vector. The operation x*v where x is a vector or
        matrix is not used.
        """
        return self*x
    ###########################################################################
    # Problem 1(h).
    ###########################################################################
    def __itruediv__(self, s):
        """
        Augmented assignment '/=' for dividing this vector with a
        scalar 's'.
        """
        for i in range(0, 3): self[i] /= s
        return self
    def __truediv__(self, s):
        """
        Return the division of this vector by a scalar 's'. The
        reflected operator, 's/v', cannot be implemented since this is
        not a defined mathematical operation.
        """
        u = +self
        u /= s
        return u
    ###########################################################################
    # Problem 1(i).
    ###########################################################################
    def __ipow__(self, i):
        """
        Augmented assignment '**=' for raising this vector to the
        integer power 'i'. For even 'i' this is a scalar and odd 'i' a
        vector.
        """
        if i < 0: raise ValueError('power must be positive')
        u = (~self)*self
        if i == 2: self = u
        # The even case.
        elif i % 2 == 0: self = u**int(i/2)
        # The odd case.
        elif i % 2 == 1: self *= u**int((i - 1)/2)
        return self
    def __pow__(self, i):
        """
        Return this vector raised to the integer power 'i'. For even
        'i' this is a scalar and odd 'i' a vector.
        """
        u = +self
        u **= i
        return u
    def __abs__(self):
        """
        Return the norm of the vector.
        """
        from math import sqrt
        return sqrt(self**2)




