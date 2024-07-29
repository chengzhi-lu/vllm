import math
import time

import random
import numpy as np
from typing import Deque
from collections import deque

from numba import njit

@njit
def fast_exp(x):
    return np.exp(x)

class SequenceGroup:
    def __init__(self, value):
         self.value= value 

def power(value, n):
     result = value ** n
    #  result = math.pow(value, n)
     return result

def sum_1(value, n):
    return np.sum(np.arange(1, n) * np.power((1 - value), np.arange(1,n)))

def sum_2(value, n):
    value = 1-value
    return value*((1-power(value,n)*(n+1-n*value))/(power((1-value),2)))

def sum_3(value,n):
     value=1-value
     return value*((1+n*value**(n+1)-(n+1)*(value**n))/(((1-value)**2)))

def sum_4(value, n):
    tmp = 0
    value=1-value
    for i in range(1, n+1):
        tmp += i * np.power(value, i)
    return tmp



print(sum_1(0.04, 16))
print(sum_2(0.04, 16))
print(sum_3(0.04, 16))
print(sum_4(0.04, 16))

def sort_by_priority(
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: sum_3(seq_group.value, 15), 
                reverse=True,
            ))

num = 500000
# 使用 set 的 union 操作
st1 = time.time()
for i in range(num):
    sum_2(random.random(),15)
    # sum_3(random.random(),15)
    # math.exp(random.random())
et1 = time.time()
print("calculate sum of ", et1-st1)

seq_groups = deque()
for i in range(num):
    seq_group= SequenceGroup(random.random())
    seq_groups.append(seq_group)
st = time.time()
sort_by_priority(seq_groups)
et = time.time()

print("sorted by priority ", et-st)

a = [random.random() for i in range(100000)]
st = time.time()
np.max(a)
et = time.time()
print("numpy max ", et-st)

st = time.time()
max(a)
et = time.time()
print("python max ", et-st)

st = time.time()
for i in range(500000):
    math.sqrt(134)
et = time.time()
print("math sqrt ", et-st)


st = time.time()
for i in range(500000):
     np.sqrt(134)
et = time.time()
print("numpy sqrt ", et-st) 

st = time.time()
a=[]
for i in range(100000):
    a.extend([random.random()])
et = time.time()
print("extend ", et-st) 

st = time.time()
a=[random.random() for i in range(100000)]
et = time.time()
print("list comprehension ", et-st) 



