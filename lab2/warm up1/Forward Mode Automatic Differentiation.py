# -*- coding: utf-8 -*-
import math

class DualNumber:
    def __init__(self, value, dvalue):
        self.value = value
        self.dvalue = dvalue

    def __str__(self):
        return str(self.value) + " + " + str(self.dvalue) + "ε"

    def __mul__(self, other):
        return DualNumber(self.value * other.value,
            self.dvalue * other.value + other.dvalue * self.value)
    
    def __add__(self, other):
        #TODO: finish me
        # YOUR CODE HERE
        return DualNumber(self.value + other.value, self.dvalue + other.dvalue)
    
    # TODO: add missing methods
    # YOUR CODE HERE
    def __sub__(self, other):
        return DualNumber(self.value - other.value, self.dvalue - other.dvalue)
    
    
    raise NotImplementedError()
    
    