# cython: language_level=3
from libcpp.vector cimport vector

def getL(a):
    cdef int N=len(a)
    cdef vector[vector[double]] m
    m.resize(N)
    for i in range(N):
        m[i].resize(N)
    for i in range(N):
        for j in range(N):
            m[i][j]=a[i][j]
    return getLambda(m)