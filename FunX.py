#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:07:08 2024

@author: alexanderpfaff
"""

from __future__ import annotations
from itertools import combinations #permutations, product
from typing import TypeVar,List, Tuple, Collection  #, Set, Dict, Optional, 
#from random import randint, shuffle 
#from math import factorial, floor, sqrt
#from copy import deepcopy
#from dataclasses import dataclass
#from collections.abc import Callable

V = TypeVar('V', str, int) 


"""     >>TOOLBOX<<  aka  'FUN (-ction)  & X (-periments)'  Module 

        This module contains (or will contain) auxiliary classes and functions 
        that supplement the Grid architecture. Notably, it contains functionalities
        that do no not implement Grid objects (-> mutual import!).
        For the most part, they are intended for arithmetic and geometric operations,
        and producing auxiliary data structures.
"""

    
# several TODOs ...
class OverflowAlgebra:
    """
        This class defines a (narural) number domain {1, 2, 3 ... {maxValue}} 
        (currently, {maxValue} allows max 25) and a simple algebra to the effect 
        that arithmetic operations producing a value v higher than {intValue} 
        will lead to 'overflow'; in other words, v will eventually be represented
        as 0 + remainder v - {maxValue} (similar to, but not identical to modulo). 
        Notice that 0 itself is not included, such that e.g. with a setting
        maxValue == 9, the addition 6 + 7 (= 13) will produce the result 4. 
        
        >>> for usage in Grid geometry 
    """
    _selectCharDomain = "123456789ABCDEFGHIJKLMNOPQ"
    def __init__(self, maxValue: int) -> None:
        if not isinstance(maxValue, int):
            raise DefinitionError("<type issue>: only int(egers) are possible input values!")
        if maxValue < 1 or maxValue > len(OverflowAlgebra._selectCharDomain):
            raise DefinitionError(f"<value issue>: the value is outside the possible range 1--{len(OverflowAlgebra._selectCharDomain)-1} \n {' ' * 16} (of chars @{OverflowAlgebra._selectCharDomain})")
        self.charRange: List[str] = OverflowAlgebra._selectCharDomain[:maxValue]
        self.intRange: List[int] = list(range(1, maxValue+1))
        self.maxInt: int = self.intRange[-1]
        self.maxChar: str = self.charRange[-1]


    def modPlus(self, intValue: int):
        out: int
        if intValue % self.maxInt == 0:
            out = self.maxInt
        else:
            out = intValue % self.maxInt 
        return out

    def toInt(self) -> int:
        pass
    

    def toVal(self) -> int:
        pass
    
    def fromInt(self, intValue: int): # -> _Value:
        toInt: int = self.modPlus(intValue, self.maxInt)
        return toInt, self.charRange[toInt-1]

            
    def __str__(self) -> str:
        strSeq: int = "["
        for char in self.charRange[:-1]:
            strSeq += char + ", "
        strSeq += self.charRange[-1] + "]"
        return f"{strSeq}"
         
    def __repr__(self):
        return f"<class 'OverFlowAlgebra' @max {self.maxInt} (= {self.maxChar}) "  

    #TODO: implement
    class _Value:
        def __init__(self, intValue: int, domain):
            if not isinstance(intValue, int):
                raise AlgebraicError("<type issue>: only int(egers) are possible input values!")
            if intValue < 1 or intValue > len(domain):
                raise AlgebraicError("<value issue>: the value is outside the possible range (1-Z)")

            self.val = domain[intValue-1]
            self.domain = domain
            self.int = intValue
            self.max = self.domain[-1]
            self.maxInt = len(self.domain)

        def __eq__(self, other):
            other = self.__jack__(other)
            return self.int == other.int
            
        def __ne__(self, other):
            other = self.__jack__(other)
            return self.int != other.int
            
        def __lt__(self, other):
            other = self.__jack__(other)
            return self.int < other.int
            
        def __le__(self, other):
            other = self.__jack__(other)
            return self.int <= other.int
            
        def __gt__(self, other):
            other = self.__jack__(other)
            return self.int > other.int
            
        def __ge__(self, other):
            other = self.__jack__(other)
            return self.int >= other.int
            

        def __add__(self, other):
            other = self.__jack__(other)

            domLen = len(self.domain)
            selfVal = self.domain.index(self.val) + 1
            otherVal = self.domain.index(other.val) + 1
            summ = selfVal + otherVal
            if summ <= domLen:
                return OverflowAlgebra._OfaNum(summ, self.domain)
            else:
                return OverflowAlgebra._OfaNum(summ - domLen, self.domain)

        def __radd__(self, other):
            other = self.__jack__(other)
            return self.__add__(other)

            
        def __sub__(self, other):
            other = self.__jack__(other)

            domLen = len(self.domain)
            selfVal = self.domain.index(self.val) + 1
            otherVal = self.domain.index(other.val) + 1
            diff = selfVal - otherVal
            if diff > 0:
                return OverflowAlgebra._OfaNum(diff, self.domain)
            else:
                return OverflowAlgebra._OfaNum(diff + domLen, self.domain)



# TODO: implement + integrate
class SquareOverflow(OverflowAlgebra):
    def __init__(self, intValue: int):
        super().__init__(intValue)
        
    def toInt(self) -> int:
        pass
    
    def toVal(self) -> int:
        pass
        
    def __str__(self) -> str:
        return "SquareOverflow @Range" + super().__str__()
    

    pass


# TODO: implement + integrate
class TraingularOverflow(OverflowAlgebra):
    def __init__(self, intValue: int):
        super().__init__(intValue)
        
    def toInt(self) -> int:
        pass
    
    def toVal(self) -> int:
        pass
        
    def __str__(self) -> str:
        return "TraingularOverflow @Range" + super().__str__()
    

    pass


class DefinitionError(Exception):
    def __init__(self, message):
        super().__init__(message)

class AlgebraicError(Exception):
    def __init__(self, message):
        super().__init__(message)









def digitSum(number: int) -> int:
    """ one way to get the exhaustive (cross-) digit sum of a given integer  """
    while len(str(number)) > 1:
        number = sum([int(y)
                 for y in str(number)])
    return number





def primeFactorize(number: int) -> Tuple[int]:
    if number < 2:
        return tuple()
    out: List[int] = []
    num_test: int = number
    divisor: int = 2
    while divisor <= num_test:
        while num_test % divisor == 0:
            out.append(divisor)
            num_test //= divisor
        divisor += 1
    return tuple(out)


def containerProduct(numbers: Collection[int]) -> int:
    product: int = 1
    for factor in numbers:
        product *= factor
    return product


def rectangularDimensions(sequence: Collection[int]):
    fullProduct = containerProduct(sequence)
    out = []
    for i in range(1, len(sequence)):
        for factorCombis in combinations(sequence, i):
            product = (containerProduct(factorCombis),
                           fullProduct // containerProduct(factorCombis))
            out.append(tuple(sorted(product)))
    return tuple(sorted(set(out)))





def pascalTriangle(rows):
    row = tuple(0
                for i in range(rows*2 + 3))
    pascalGrid = [list(row)
                  for i in range(rows + 1)]
    middle = rows + 1
    pascalGrid[0][middle] = 1
    for deep in range(1, rows + 1):
        for wide in range(middle - deep, middle + deep + 1, 2):
            pascalGrid[deep][wide] = pascalGrid[deep-1][wide-1] + pascalGrid[deep-1][wide+1]
    return pascalGrid


# p = pascalTriangle(20)
# for r in p:
#     for c in r:
#         if c == 0:
#             print("     ", end="")
#         else:
#             print(f"{c:5d}",  end="")
#     print()























