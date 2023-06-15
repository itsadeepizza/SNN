import numpy as np

def update_weights(W, S_pre, S_post, rk, learning_rate, aP_plus, aP_minus):
    # Calculate the time difference between pre and post spikes
    time_diff = S_post[:, np.newaxis] - S_pre

    # Apply the conditions for updating the weights
    positive_diff_mask = time_diff > 0
    negative_diff_mask = ~positive_diff_mask

    # Calculate the exponential factors
    exp_factors = np.exp(-np.abs(time_diff) / 10)

    # Calculate the weight updates
    positive_updates = aP_plus * W.T * (1 - W.T) * learning_rate * exp_factors * rk
    negative_updates = aP_minus * W.T * (1 - W.T) * learning_rate * exp_factors * rk

    # Apply the updates to the weights
    W += np.sum(positive_diff_mask * positive_updates, axis=-1).T
    W -= np.sum(negative_diff_mask * negative_updates, axis=-1).T

    return W
