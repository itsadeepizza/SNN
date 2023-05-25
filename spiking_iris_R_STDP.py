import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

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
        return np.where(self.delays == t, 1, 0)


class SpikingLayerLeaky:
    def __init__(self, input_dim, output_dim, alpha=0.5, beta=0.5, theta=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.V = np.random.randn(input_dim, output_dim)
        # Initialize self.W using a beta distribution
        self.W = np.random.beta(1, 1, (output_dim, input_dim))
        self.I = np.zeros(output_dim)
        self.U = np.zeros(output_dim)
        self.S = np.zeros(output_dim)
        # time
        self.t = 0

        self.alpha = alpha
        self.beta = beta
        self.theta = theta

        self.S_history = [[]] * output_dim

    def forward(self, input):
        self.U = self.beta * (self.U + self.I - self.S)
        self.I = self.alpha * self.I + self.W @ input  # + self.V @ input
        self.S = np.where(self.U > self.theta, 1, 0)
        # Record spikes
        for i in range(self.output_dim):
            if self.S[i] == 1:
                self.S_history[i].append(self.t)
        # Update time
        self.t += 1
        return self.S

    def reset(self):
        self.I = np.zeros(self.output_dim)
        self.U = np.zeros(self.output_dim)
        self.S = np.zeros(self.output_dim)
        self.S_history = [[]] * output_dim
        self.t = 0


# HYPERPARAMETERS
N = 25  # Number of neurons for features
sigma = 0.5  # variance of gaussians
threshold = 0.01  # firing threshold for gaussians
T = 100  # Number of time steps
n_features = 4
input_dim = N * n_features
output_dim = 3

# INIT LAYERS
# Initialize encoders
sepal_length_encoder = SpikingEncoder(x_min=4, x_max=8, sigma=sigma, threshold=threshold, T=T, N=N)
sepal_width_encoder = SpikingEncoder(x_min=2, x_max=4.5, sigma=sigma, threshold=threshold, T=T, N=N)
petal_length_encoder = SpikingEncoder(x_min=1, x_max=7, sigma=sigma, threshold=threshold, T=T, N=N)
petal_width_encoder = SpikingEncoder(x_min=0, x_max=2.5, sigma=sigma, threshold=threshold, T=T, N=N)

# Initialize output layer
layer = SpikingLayerLeaky(input_dim, output_dim)

# Load iris dataset and plot distribution for each feature
dataset = pd.read_csv('iris.csv')
# shuffle dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)
print(dataset.head())


# Extract sample from dataset
for epoch in range(10000):
    # Make a heatmap from the weights

    plt.imshow(layer.W, cmap='gray', interpolation='nearest')
    plt.show()
    total_reward = 0
    for n_sample in tqdm(range(len(dataset))):
        layer.reset()
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
            input = np.concatenate((sepal_length_encoder.get_state(t), sepal_width_encoder.get_state(t), petal_length_encoder.get_state(t), petal_width_encoder.get_state(t)))
            output = layer.forward(input)
            output_memory += output
        # Take max index of output as prediction
        pred = np.argmax(output_memory)
        # Calculate reward for output layer
        rk = np.zeros(output_dim)
        if pred == target:
            # Reward correct prediction are all zeros except for the correct prediction
            rk[pred] = 1
            total_reward += 1
        else:
            # Reward incorrect prediction: +1 for correct target, -1 for incorrect prediction
            rk[pred] = -1
            rk[target] = 1
        S_pre = [*sepal_length_encoder.S_history,
                 *sepal_width_encoder.S_history,
                 *petal_length_encoder.S_history,
                 *petal_width_encoder.S_history]
        S_post = layer.S_history

        aP_plus = 1
        aP_minus = 1
        learning_rate = 1e-4

        for i in range(len(S_pre)):
            for j in range(len(S_post)):
                # cached_prod = layer.W[j, i] * ( 1 - layer.W[j, i])
                for pre_spike in S_pre[i]:
                    for post_spike in S_post[j]:
                        if post_spike - pre_spike > 0:
                            layer.W[j, i] += aP_plus * layer.W[j, i] * ( 1 - layer.W[j, i]) * learning_rate
                        else:
                            layer.W[j, i] -= aP_minus * layer.W[j, i] * ( 1 - layer.W[j, i]) * learning_rate

        # print(rk)

    print("Epoch: ", epoch, "Reward: ", total_reward / len(dataset))

