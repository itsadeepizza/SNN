import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import slowspike
import fastspike
import matspike
import random

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

class SpikingEncoder:
    def __init__(self, x_min, x_max, sigma, threshold, T, N=25):
        self.x_min = x_min
        self.x_max = x_max
        self.sigma = sigma
        self.threshold = threshold
        self.T = T
        self.N = N
        self.mu = np.linspace(x_min, x_max, N)


    def gaussian(self, x):
        return gaussian(np.ones(self.N) * x, self.mu, np.ones(self.N) * self.sigma)


    def calculate_delays(self, x):
        # Calculate delays for a given input x
        delays = self.gaussian(x)
        delays[delays < threshold] = -1
        self.delays = np.rint((1 - delays) * self.T).astype(int)
        self.S_history = [[self.delays[i]] if delays[i] != -1 else [] for i in range(self.N)]

    def get_state(self, t):
        # Return state of neurons at time t
        return (self.delays == t).astype(int)


class SpikingLayerLeaky:
    def __init__(self, input_dim, output_dim, alpha=0.5, beta=0.5, theta=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.V = np.random.randn(input_dim, output_dim)
        # Initialize self.W using a beta distribution
        self.W = np.random.beta(1, 1, (output_dim, input_dim)) # Weights
        self.I = np.zeros(output_dim) # Current
        self.U = np.zeros(output_dim) # Potential
        self.S = np.zeros(output_dim) # Spike (output)
        # time
        self.t = 0

        self.alpha = alpha # Decay for current
        self.beta = beta # Decay for potential
        self.theta = theta # Threshold

        self.S_history = [[]] * self.output_dim

    def forward(self, input):
        self.U = self.beta * (self.U + self.I - self.S)
        self.I = self.alpha * self.I + self.W @ input  # + self.V @ input
        self.S = (self.U > self.theta).astype(int)

        # Record spikes
        for i in range(self.output_dim):
            if self.S[i] == 1:
                # print(self.S_history[i])
                self.S_history[i].append(self.t)

        # Update time
        self.t += 1

        return self.S

    def backward(self, rk_out):
        rk_inp = np.abs(self.W).T @ rk_out

        return rk_inp

    def reset(self):
        self.I = np.zeros(self.output_dim)
        self.U = np.zeros(self.output_dim)
        self.S = np.zeros(self.output_dim)
        self.S_history = [[]] * self.output_dim
        self.t = 0


# HYPERPARAMETERS
N = 25  # Number of neurons for features
sigma = 0.5  # variance of gaussians
threshold = 0.01  # firing threshold for gaussians
T = 100  # Number of time steps
n_features = 4
input_dim = N * n_features
hidden_dim = 8
output_dim = 3

# INIT LAYERS
# Initialize encoders
sepal_length_encoder = SpikingEncoder(x_min=4, x_max=8, sigma=sigma, threshold=threshold, T=T, N=N)
sepal_width_encoder = SpikingEncoder(x_min=2, x_max=4.5, sigma=sigma, threshold=threshold, T=T, N=N)
petal_length_encoder = SpikingEncoder(x_min=1, x_max=7, sigma=sigma, threshold=threshold, T=T, N=N)
petal_width_encoder = SpikingEncoder(x_min=0, x_max=2.5, sigma=sigma, threshold=threshold, T=T, N=N)

# Initialize output and hidden layer
hidden_layer = SpikingLayerLeaky(input_dim, hidden_dim)
output_layer = SpikingLayerLeaky(hidden_dim, output_dim)

# Load iris dataset and plot distribution for each feature
dataset = pd.read_csv('iris.csv')
# split into train and test
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)

print(dataset.head())

# Extract sample from dataset
for epoch in range(10000):
    # Make a heatmap from the weights
    if epoch % 2 == 0:
        mode = "train"
        # Shuffle dataset
        dataset = train.sample(frac=1)
    else:
        mode = "test"
        dataset = test.sample(frac=1)

    # plt.imshow(layer.W, cmap='gray', interpolation='nearest')
    # plt.show()
    total_reward = 0
    for n_sample in tqdm(range(len(dataset))):
        hidden_layer.reset()
        output_layer.reset()

        sample = dataset.iloc[n_sample]

        # Encode sample
        sepal_length_encoder.calculate_delays(sample["sepal.length"])
        sepal_width_encoder.calculate_delays(sample["sepal.width"])
        petal_length_encoder.calculate_delays(sample["petal.length"])
        petal_width_encoder.calculate_delays(sample["petal.width"])

        varieties = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}

        target = varieties[sample["variety"]]

        # RUN NETWORK on sample for T time steps
        U_by_time = []
        output_memory = 0

        for t in range(T):
            hidden_input = np.concatenate((sepal_length_encoder.get_state(t), sepal_width_encoder.get_state(t), petal_length_encoder.get_state(t), petal_width_encoder.get_state(t)))

            middle_output = hidden_layer.forward(hidden_input)
            output = output_layer.forward(middle_output)
            output_memory += output

        # Take max index of output as prediction
        pred = np.argmax(output_memory)
        # Calculate reward for output layer
        r_out = np.zeros(output_dim)

        if pred == target:
            # Reward correct prediction are all zeros except for the correct prediction
            r_out[pred] = 1
            total_reward += 1
        else:
            # Reward incorrect prediction: +1 for correct target, -1 for incorrect prediction
            r_out[pred] = -1
            # r_out[target] = 1

        S_input = [*sepal_length_encoder.S_history,
                   *sepal_width_encoder.S_history,
                   *petal_length_encoder.S_history,
                   *petal_width_encoder.S_history]

        S_hidden = hidden_layer.S_history
        S_output = output_layer.S_history

        aP_plus = 1.5
        aP_minus = 1
        learning_rate = 1e-4
        if mode == "train":
            # hidden_layer.W = fastspike.update_weights(hidden_layer.W, S_input, S_hidden, output_layer.backward(r_out), learning_rate, aP_plus, aP_minus)
            # output_layer.W = fastspike.update_weights(output_layer.W, S_hidden, S_output, r_out, learning_rate, aP_plus, aP_minus)
            if random.random() < 0.5:
                output_layer.W = fastspike.update_weights(output_layer.W, S_hidden, S_output, r_out, learning_rate, aP_plus, aP_minus)
            else:
                hidden_layer.W = fastspike.update_weights(hidden_layer.W, S_input, S_hidden, output_layer.backward(r_out), learning_rate, aP_plus, aP_minus)


    print("Epoch: ", epoch // 2, "Reward: ", total_reward / len(dataset), "Mode: ", mode)
