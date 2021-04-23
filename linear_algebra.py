#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from functools import partial, reduce


# In[2]:


#
# Functions for working with vectors
#


# In[3]:


def vector_add(v,w):
    '''Soma elementos correspondentes'''
    return [v_i + w_i
           for v_i, w_i in zip(v,w)]


# In[4]:


def vector_subtract(v,w):
    '''Seubtrai elementos correspondentes'''
    return [v_i - w_i
           for v_i, w_i in zip(v,w)]


# In[5]:


'''Cria um vetor cujo o primeiro elemento seja a soma de todos os primeiros elementos'''
def vector_sum(vectors):
    return reduce(vector_add, vectors)


# In[6]:


'''Simplesmente faz ele multiplicar cada valor de V por '''
def scalar_multiply(c,v):
    '''c é um numero, v é um vetor'''
    return [c * v_i for v_i in v]


# In[7]:


'''Computar a média de uma lista de vetores'''

def vector_mean(vectors):
    """Computar o vetor cujo i-ésimo elemento seja a média dos 
    i-ésimos elementos dos vetores inclusos"""
    
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


# In[8]:


'''O produto escalar de dois vetores é a soma de seus
produtos componente a componente'''

def dot(v,w):
    '''v_1 * w_1 + ... + v_n * w_n '''
    return sum(v_i * w_i
              for v_i, w_i in zip(v,w))


# In[9]:


'''Soma dos quadrados de um vetor'''

def sum_of_squares(v):
    '''v_1 * v_1 + ... + v_n *v_n'''
    
    return dot(v,v)


# In[10]:


'''Calcular a magnitude (ou tamanho)'''
def magnitude(v):
    return math.sqrt(sum_of_squares(v)) # math.sqrt é a função da raiz quadrada


# In[11]:


'''Calcular a distancia entre dois vetores'''
def squared_distance(v,w):
    '''(v_1 * w_1) ** 2 + ... + (v_n * w_n) ** 2'''
    
    return sum_of_squares(vector_subtract(v,w))

def distance(v,w):
    return math.sqrt(squared_distance(v,w))


# In[12]:


#
# functions for working with matrices
#


# In[13]:


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


# In[14]:


def get_row(A, i):
    return A[i]          # A[i] já é da linha A[i] é linha i-ésimo

def get_column(A, j):
    return [A_i[j]       # j-ésimo elemento da linha A_i
           for A_i in A] # para cada linha A_i


# In[15]:


def make_matrix(num_rows, num_cols, entry_fn):
    '''retorna a matriz num_rows X num_cols
    cuja entrada (i,j)th é entry_fn(i,j)'''
    
    return [[entry_fn(i,j)            # dado i, cria uma lista
             for j in range(num_cols)]# [entry_fn(i,0), ...]
           for i in range(num_rows)]  # cria uma lista para cada i


# In[16]:


def is_diagonal(i, j):
    """1's na diagonal, 0's nos demais lugares"""
    
    return 1 if i == j else 0

indentity_matrix = make_matrix(5,5,is_diagonal)
print(f'{indentity_matrix}')


# In[ ]:




