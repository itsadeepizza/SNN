import numpy as np
import pandas as pd
from tqdm import tqdm

# ╔╦╗╔═╗╔╦╗╔═╗╦
# ║║║║ ║ ║║║╣ ║
# ╩ ╩╚═╝═╩╝╚═╝╩═╝

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

class SNN:
    pass

class SpikingEncoder(SNN):
    """
    Encode a scalar value into a spike train using gaussian functions
    Cit: "Gaussian receptive field"
    """
    def __init__(self, x_min, x_max, sigma, threshold, T, N=25):
        self.x_min = x_min
        self.x_max = x_max
        self.sigma = sigma
        self.threshold = threshold
        self.T = T
        self.N = N
        self.output_dim = N
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
        spikes = (self.delays == t).astype(int)
        return spikes


class MultiEncoder(SNN):
    def __init__(self, encoders):
        self.encoders = encoders
        self.output_dim = sum([encoder.output_dim for encoder in encoders])
        # Last spikes
        self.recorded_spikes = np.zeros((epsilon, self.output_dim))

    def calculate_delays(self, x):
        for encoder, x_i in zip(self.encoders, x):
            encoder.calculate_delays(x_i)

    def get_state(self, t):
        spikes = np.concatenate([encoder.get_state(t) for encoder in self.encoders])
        # Record spikes (all recorded spikes roll in array and the oldest are replaced by last spikes)
        self.recorded_spikes = np.roll(self.recorded_spikes, -1, axis=0)
        self.recorded_spikes[-1, :] = spikes
        return spikes



class SpikingLayer(SNN):
    """
    Spiking layer for BP-STDP (https://arxiv.org/pdf/1711.04214.pdf)
    """
    def __init__(self, prev_layer:SNN, output_dim:int, theta=0.9):
        self.prev_layer = prev_layer
        self.input_dim = self.prev_layer.output_dim
        self.output_dim = output_dim
        # Initialize self.W using a beta distribution as it has to be positive
        self.W = np.random.beta(1, 1, (output_dim, self.input_dim)) # Weights
        self.U = np.zeros(output_dim) # Potential
        self.S = np.zeros(output_dim) # Spike (output)

        # Last spikes
        self.recorded_spikes = np.zeros((epsilon, output_dim))
        # time
        self.t = 0

        self.theta = np.float32(theta) # Threshold


    def forward(self, input):
        self.U = self.U + self.W @ input - self.S
        self.S = (self.U > self.theta).astype(int)
        # Record spikes (all recorded spikes roll in array and the oldest are replaced by last spikes)
        self.recorded_spikes = np.roll(self.recorded_spikes, -1, axis=0)
        self.recorded_spikes[-1] = self.S
        # Reset potential for neurons that spiked
        self.U = self.U * (self.S == 0)

        # Update time
        self.t += 1
        return self.S

    def backward(self, reward, get_reward=True):
        """
        Update weights using BP-STDP
        :param reward: this is called csi in the paper for the output layer. We generalize it to all layers
        """
        # Add a dummy dimension to csi
        delta_W = reward[:, None] @ np.sum(self.prev_layer.recorded_spikes, axis=0)[None, :]
        # Backward pass of the reward
        if get_reward:
            new_reward = np.sum(self.W * delta_W, axis=0)
        else:
            new_reward = None
        # Update weights
        self.W += learning_rate * delta_W
        return new_reward

    def get_csi(self, label):
        """
        Compute csi for the output layer (formula 16 in the paper)
        :param label: one hot encoded labels
        :return: csi
        """
        r = np.max(self.recorded_spikes, axis=0)
        csi = label - r
        return csi

    def reset(self):
        self.U.fill(0)
        self.S.fill(0)
        self.recorded_spikes.fill(0)
        self.t = 0

# ╦ ╦╦ ╦╔═╗╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔╦╗╔═╗╔╦╗╔═╗╦═╗╔═╗
# ╠═╣╚╦╝╠═╝║╣ ╠╦╝╠═╝╠═╣╠╦╝╠═╣║║║║╣  ║ ║╣ ╠╦╝╚═╗
# ╩ ╩ ╩ ╩  ╚═╝╩╚═╩  ╩ ╩╩╚═╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═╚═╝

N = 25  # Number of neurons for features
sigma = 0.5  # variance of gaussians
threshold = 0.01  # firing threshold for gaussians
T = 100  # Number of time steps
n_features = 4
input_dim = N * n_features
hidden_dim = 20
output_dim = 3
learning_rate = 0.1
epsilon = 4

# ╦ ╔╗╔ ╦ ╔╦╗
# ║ ║║║ ║  ║
# ╩ ╝╚╝ ╩  ╩



# Initialize encoders
sepal_length_encoder = SpikingEncoder(x_min=4, x_max=8, sigma=sigma, threshold=threshold, T=T, N=N)
sepal_width_encoder = SpikingEncoder(x_min=2, x_max=4.5, sigma=sigma, threshold=threshold, T=T, N=N)
petal_length_encoder = SpikingEncoder(x_min=1, x_max=7, sigma=sigma, threshold=threshold, T=T, N=N)
petal_width_encoder = SpikingEncoder(x_min=0, x_max=2.5, sigma=sigma, threshold=threshold, T=T, N=N)

multi_encoder = MultiEncoder([sepal_length_encoder, sepal_width_encoder, petal_length_encoder, petal_width_encoder])
# Initialize output and hidden layer
hidden_layer = SpikingLayer(multi_encoder, hidden_dim, theta=0.9)
output_layer = SpikingLayer(hidden_layer, output_dim, theta=5)

# Load iris dataset and plot distribution for each feature
dataset = pd.read_csv('dataset/iris.csv')
# split into train and test
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)
varieties = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
# ╔╦╗╦═╗╔═╗╦╔╗╔
#  ║ ╠╦╝╠═╣║║║║
#  ╩ ╩╚═╩ ╩╩╝╚╝

for epoch in range(10000):
    # Make a heatmap from the weights
    if epoch % 2 == 1:
        mode = "train"
        # Shuffle dataset
        dataset = train.sample(frac=1)
    else:
        mode = "test"
        dataset = test.sample(frac=1)

    # plt.imshow(layer.W, cmap='gray', interpolation='nearest')
    # plt.show()
    total_reward = 0

    correct_predictions = 0
    for n_sample in tqdm(range(len(dataset))):
        hidden_layer.reset()
        output_layer.reset()

        sample = dataset.iloc[n_sample]
        # Encode sample
        multi_encoder.calculate_delays([sample['sepal.length'], sample['sepal.width'], sample['petal.length'], sample['petal.width']])
        target = varieties[sample["variety"]]

        # Generate label as one hot encoding
        label = np.zeros(output_dim)
        label[target] = 1
        total_output = np.zeros(output_dim)
        for t in range(T):
            encoded_sample = multi_encoder.get_state(t)
            middle_output = hidden_layer.forward(encoded_sample)
            output = output_layer.forward(middle_output)
            total_output += output

            if mode == "train" and t % epsilon == 0:
                csi = output_layer.get_csi(label)
                reward = output_layer.backward(csi)
                reward = hidden_layer.backward(reward)

        # Check if prediction is correct
        pred = np.argmax(total_output)
        if pred == target:
            correct_predictions += 1
    accuracy = correct_predictions / len(dataset)
    print()
    print(f"EPOCH {epoch//2} - {mode} - ACCURACY: {accuracy}")


