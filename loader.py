import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

class SpikingEncoder:
    def __init__(self, x, x_min, x_max, sigma, threshold, T, N=25):
        def gaussian(x, mu, sigma):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

        self.x_min = x_min
        self.x_max = x_max
        self.sigma = sigma
        self.threshold = threshold
        self.T = T
        self.N = N

        mu = np.linspace(x_min, x_max, N)
        delays = gaussian(np.ones(N) * x, mu, np.ones(N) * sigma)
        delays[delays < threshold] = np.NAN
        self.delays = np.rint((1 - delays) * T).astype(int)

    def get_state(self, t):
        # Return state of neurons at time t
        return np.where(self.delays == t, 1, 0)

N = 25 # Number of neurons for features
sigma = 0.5 # variance of gaussians
threshold = 0.01 # firing threshold for gaussians
T = 100 # Number of time steps


i = 5

sample = dataset.iloc[i]


sepal_length_encoder = SpikingEncoder(x=sample["sepal.length"], x_min=4, x_max=8, sigma=sigma, threshold=threshold, T=T, N=N)
sepal_width_encoder = SpikingEncoder(x=sample["sepal.width"], x_min=2, x_max=4.5, sigma=sigma, threshold=threshold, T=T, N=N)
petal_length_encoder = SpikingEncoder(x=sample["petal.length"], x_min=1, x_max=7, sigma=sigma, threshold=threshold, T=T, N=N)
petal_width_encoder = SpikingEncoder(x=sample["petal.width"], x_min=0, x_max=2.5, sigma=sigma, threshold=threshold, T=T, N=N)

for t in range(T):
    input = np.concatenate((sepal_length_encoder.get_state(t), sepal_width_encoder.get_state(t), petal_length_encoder.get_state(t), petal_width_encoder.get_state(t)))
    print(input)
