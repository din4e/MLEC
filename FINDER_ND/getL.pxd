# cython: language_level=3
from libcpp.vector cimport vector

cdef extern from "./src/lib/getL.h":
    double getLambda(vector[vector[double]] &a)
