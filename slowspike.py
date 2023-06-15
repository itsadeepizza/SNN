import numpy as np

def update_weights(W, S_pre, S_post, rk, learning_rate, aP_plus, aP_minus):
    for i in range(len(S_pre)):
        for j in range(len(S_post)):
            # cached_prod = layer.W[j, i] * (1 - layer.W[j, i])
            for pre_spike in S_pre[i]:
                for post_spike in S_post[j]:
                    # sign = (-1 if post_spike - pre_spike < 0 else 1)
                    # W[j, i] += [aP_plus if sign == 1 else aP_minus] * W[j, i] * (1 - W[j, i]) * learning_rate * np.exp(-(post_spike - pre_spike) / 10) * rk[j] * (-1 if post_spike - pre_spike < 0 else 1)
                    if post_spike - pre_spike > 0:
                        W[j, i] += aP_plus * W[j, i] * (1 - W[j, i]) * learning_rate * np.exp(-(post_spike - pre_spike) / 10) * rk[j]
                    else:
                        W[j, i] -= aP_minus * W[j, i] * (1 - W[j, i]) * learning_rate * np.exp(-(pre_spike - post_spike) / 10) * rk[j]

    return W