#%%

import numpy             as np
import matplotlib.pyplot as plt


np.set_printoptions(precision = 4)

#%%

NUM_SAMPLES  = 10
NUM_FEATURES = 4
NUM_NEURONS  = 10
NUM_POINTS   = NUM_NEURONS*20

#%%


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

N = 25 # Number of neurons for features
x_min = 1
x_max = 3
sigma = 0.5 # variance of gaussians
threshold = 0.01 # firing threshold for gaussians
T = 100 # Number of time steps

x = 1.3
mu = np.linspace(x_min, x_max, N)
delays = (1 - gaussian(np.ones(N)*x, mu, np.ones(N)*sigma))
delays = np.rint(delays * T).astype(int)

i = 3 # time instant
np.where(delays == i, 1, 0)


print(delays)

fig, ax = plt.subplots(2)
ax[0].scatter(range(N), delays)
ax[0].set_title("Delays")
ax[0].set_xlabel("Time (ms)")
ax[0].set_ylabel("Neuron index")
# Plot all the gaussians
for i in range(N):
    ax[1].plot(np.linspace(x_min, x_max, 100), gaussian(np.linspace(x_min, x_max, 100), mu[i], sigma))

    # Mark point (x, delays[i])
    ax[1].scatter(x, gaussian(x, mu[i], sigma))
plt.show()

# layer 0 INPUT LAYER
# -----------
# layer 1 OUTPUT LAYER - a neuron for each feature
# -----------



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
        self.U = self.beta * self.U + self.I - self.S
        self.I = self.alpha * self.I + self.W @ input # + self.V @ input
        self.S = np.where(self.U > self.theta, 1, 0)
        return self.S

n_features = 1
input_dim = N * n_features
output_dim = 3

layer = SpikingLayerLeaky(input_dim, output_dim)

U_by_time = []
for t in range(T):
    input = np.where(delays == t, 1, 0)
    output = layer.forward(input)
    U_by_time.append(layer.U)
    print(output)
U_by_time = np.array(U_by_time)
fig, ax = plt.subplots()
for i in range(U_by_time.shape[1]):
    ax.plot(range(U_by_time.shape[0]), U_by_time[:,i])
plt.show()
