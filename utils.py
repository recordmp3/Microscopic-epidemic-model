import random
import numpy as np
import math
def shuffle(a):
    np.random.shuffle(a)
    return a
 
def hpn(p):
    if randreal() <= p: return True
    return False

def onehot(pos, length):

    return [1 if i == pos else 0 for i in range(length)]

def m_sample(p_l):  # sampling a multinomial distribution
 
    p = randreal()
    for i in range(len(p_l)):
        if p < p_l[i]: return i
        p -= p_l[i]
 

def randint1(a):
    #x = randint(0, a)
    #return x
    return np.random.randint(a)
def randint(a, b):
    #x = a + math.floor((b - a - 1) * randreal() + 0.5)
    #return x
    return np.random.randint(a, b)
def Rand(a, b):
    return a + (b - a) * randreal()

seed, begin, bias = 19260817, 1000000007, 233737

def randreal():
    # global begin
    #begin = (begin * seed + bias) % 1000000009
    # print(begin)
    # return begin / 1000000009
    return np.random.random() 
