import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        self.delays = np.rint((1 - delays) * T).astype(int)

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

        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, input):
        self.U = self.beta * (self.U + self.I - self.S)
        self.I = self.alpha * self.I + self.W @ input  # + self.V @ input
        self.S = np.where(self.U > self.theta, 1, 0)
        return self.S


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
print(dataset.head())
# plot distribution for each feature
fig, ax = plt.subplots(4, figsize=(10, 10))
ax[0].hist(dataset["sepal.length"])
ax[0].set_title("Sepal Length")
ax[1].hist(dataset["sepal.width"])
ax[1].set_title("Sepal Width")
ax[2].hist(dataset["petal.length"])
ax[2].set_title("Petal Length")
ax[3].hist(dataset["petal.width"])
ax[3].set_title("Petal Width")
plt.show()

# Extract sample from dataset
i = 5
sample = dataset.iloc[i]

# Encode sample
sepal_length_encoder.calculate_delays(sample["sepal.length"])
sepal_width_encoder.calculate_delays(sample["sepal.width"])
petal_length_encoder.calculate_delays(sample["petal.length"])
petal_width_encoder.calculate_delays(sample["petal.width"])

# Plot delays
fig, ax = plt.subplots(2, figsize=(8, 8))
ax[0].scatter(range(N), sepal_length_encoder.delays)
ax[0].set_title("Delays")
ax[0].set_xlabel("Time (ms)")
ax[0].set_ylabel("Neuron index")
# Plot all the gaussians
Xs = np.linspace(sepal_length_encoder.x_min, sepal_length_encoder.x_max, 100)
for i in range(N):
    ax[1].plot(Xs, gaussian(Xs, sepal_length_encoder.mu[i], sepal_length_encoder.sigma))
    # Mark point (x, delays[i])
    ax[1].scatter(sample["sepal.length"], sepal_length_encoder.gaussian(sample["sepal.length"])[i], c="red")
plt.show()

# RUN NETWORK on sample for T time steps
U_by_time = []
for t in range(T):
    input = np.concatenate((sepal_length_encoder.get_state(t), sepal_width_encoder.get_state(t), petal_length_encoder.get_state(t), petal_width_encoder.get_state(t)))
    output = layer.forward(input)
    print(output)

    U_by_time.append(layer.U)

# Plot time evolution of membrane potentials
U_by_time = np.array(U_by_time)
fig, ax = plt.subplots()
for i in range(U_by_time.shape[1]):
    ax.plot(range(U_by_time.shape[0]), U_by_time[:,i])
plt.show()