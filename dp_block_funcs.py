import numpy as np
import random
import math
import pandas as pd


def calc_pB_block(epsilon, blocks):
    B = len(blocks)
    pB = math.exp(epsilon)/(B-1+math.exp(epsilon))
    return pB


def calc_pB_individual(epsilon, blocks, values):
    n=max(values)
    m=max(len(x) for x in blocks)
    pB = math.exp(epsilon)/(math.exp(epsilon)+n/m-1)
    return pB


def reportNoisyMax(values, epsilon):
    '''
    Report differentially private noisy max - > from Section3: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    '''
    rVals = list()
    for val in values:
        rval = val + np.random.laplace(0,2.0/epsilon)
        rVals.append(rval)
    
    #print(rVals)
    max_value = max(rVals)
    max_index = rVals.index(max_value)
    noisy_max = values[max_index] + np.random.laplace(0,2.0/epsilon)
    return noisy_max
    

def blockRR(value, blocks, pB):
    '''
    Perform block/interval randomized response
    with prob pB the original block is kept, (1-pB) we choose a random block
    '''
    
    valB = None
    for i in range(0, len(blocks)):
        if value in blocks[i]:
            valB = i
            break
            
    if valB is None:
        valB = len(blocks)-1
    
    #print(str(value)+" "+str(valB))
    p = random.random()
    
    if p<=pB:
        privVal = random.choice(blocks[valB])
    else:
        otherVal = list()
        for j in range(0, len(blocks)):
            if j!=valB:
                otherVal.extend(blocks[j])
                
        privVal = random.choice(otherVal)
        
    return privVal
    
    
def sanitizeActProfile(IDs, individualAct, pB, blocks, memo=False):
    allPrivAct = dict()
    for Id in IDs:
        privAct = list()
        privmap = dict()
        
        if memo:
            for val in set(individualAct[Id]):
                pval = blockRR(val, blocks, pB)
                privmap[val] = pval

            for val in individualAct[Id]:
                privAct.append(privmap[val])
        else:
            for val in individualAct[Id]:
                pval = blockRR(val, blocks, pB)
                privAct.append(pval)
                
        allPrivAct[Id]=privAct
    return allPrivAct