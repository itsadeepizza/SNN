cimport numpy as cnp
from libc.math cimport exp

cpdef cnp.ndarray[double, ndim=2] update_weights(
    cnp.ndarray[double, ndim=2] W,
    list S_pre,
    list S_post,
    cnp.ndarray[double, ndim=1] rk,
    double learning_rate,
    double aP_plus,
    double aP_minus
):
    cdef int i, j
    cdef double pre_spike, post_spike

    for i in range(len(S_pre)):
        for j in range(len(S_post)):
            for pre_spike in S_pre[i]:
                for post_spike in S_post[j]:
                    if post_spike - pre_spike > 0:
                        W[j, i] += aP_plus * W[j, i] * (1 - W[j, i]) * learning_rate * exp(-(post_spike - pre_spike) / 10) * rk[j]
                    else:
                        W[j, i] -= aP_minus * W[j, i] * (1 - W[j, i]) * learning_rate * exp(-(pre_spike - post_spike) / 10) * rk[j]
    return W

cpdef cnp.ndarray[float, ndim=2] get_delta_W(
    cnp.ndarray[float, ndim=2] W,
    list S_pre,
    list S_post,
    cnp.ndarray[float, ndim=1] rk,
    float learning_rate,
    float aP_plus,
    float aP_minus
):
    cdef int i, j
    cdef float pre_spike, post_spike
    cdef cnp.ndarray[float, ndim=2] delta_W
    delta_W = W * 0

    for i in range(len(S_pre)):
        for j in range(len(S_post)):
            for pre_spike in S_pre[i]:
                for post_spike in S_post[j]:
                    if post_spike - pre_spike > 0:
                        delta_W[j, i] += aP_plus * W[j, i] * (1 - W[j, i]) * learning_rate * exp(-(post_spike - pre_spike) / 10) * rk[j]
                    else:
                        delta_W[j, i] -= aP_minus * W[j, i] * (1 - W[j, i]) * learning_rate * exp(-(pre_spike - post_spike) / 10) * rk[j]
    return delta_W
