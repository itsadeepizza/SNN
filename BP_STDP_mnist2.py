import numpy as np
import pandas as pd
from tqdm import tqdm
from mnist_encoder3 import SpikingDataset, SpikingImageEncoder
from matplotlib import pyplot as plt

# ╔╦╗╔═╗╔╦╗╔═╗╦
# ║║║║ ║ ║║║╣ ║
# ╩ ╩╚═╝═╩╝╚═╝╩═╝


class SNN:
    pass


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
        # TODO: check this
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


threshold = 0.01  # firing threshold for gaussians
T = 100  # Number of time steps

hidden_dim = 100
output_dim = 10
learning_rate = 0.005
epsilon = 4
theta_0 = 0.9
theta_1 = 5

def train():
    # ╦ ╔╗╔ ╦ ╔╦╗
    # ║ ║║║ ║  ║
    # ╩ ╝╚╝ ╩  ╩
    # Initialize output and hidden layer
    encoder_layer = SpikingImageEncoder(28, 28, T, epsilon=epsilon)
    hidden_layer = SpikingLayer(encoder_layer, hidden_dim, theta=theta_0)
    output_layer = SpikingLayer(hidden_layer, output_dim, theta=theta_1)

    spiking_dataset = SpikingDataset(("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte"), T=T)
    # ╔╦╗╦═╗╔═╗╦╔╗╔
    #  ║ ╠╦╝╠═╣║║║║
    #  ╩ ╩╚═╩ ╩╩╝╚╝

    all_rows = list(range(len(spiking_dataset)))
    train = np.random.choice(all_rows, int(0.7*len(all_rows)), replace=False)
    test = np.array([i for i in all_rows if i not in train])
    train_losses = []
    test_losses = []

    mode = "train"
    for epoch in range(50):
        if epoch % 2 == 0:
            mode = "test"
        else:
            mode = "train"
        total_reward = 0

        correct_predictions = 0
        if mode == "train":
            subset = np.random.choice(train, 1000, replace=False)
        else:
            subset = np.random.choice(test, 300, replace=False)

        # for i,(image_encode, label) in enumerate(tqdm(spiking_dataset)):
        for i in tqdm(subset):
            image_encode, label = spiking_dataset[i]
            hidden_layer.reset()
            output_layer.reset()
            encoder_layer.load_bins(image_encode)
            target = np.argmax(label)

            total_output = np.zeros(output_dim)
            for t in range(T):
                encoded_sample = encoder_layer.get_state(t)
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
        accuracy = correct_predictions / len(subset)
        print()
        print(f"EPOCH {epoch // 2} - {mode} - ACCURACY: {accuracy}")
        if mode == "train":
            train_losses.append(accuracy)
        else:
            test_losses.append(accuracy)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(train_losses, label="train")
    ax.plot(test_losses, label="test")
    ax.legend()
    # Write an annotation with the hypermaters
    ax.annotate(f"epsilon={epsilon},\n hidden_dim={hidden_dim},\n  theta0={theta_0},\n theta1={theta_1}\n learning_rate={learning_rate}",
                xy=(0.5, 1), xycoords='axes fraction',
                xytext=(0.5, 1.25), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='top',
                )
    # Fit size
    fig.tight_layout()
    plt.savefig(f"plots/epsilon={epsilon}, hidden_dim={hidden_dim},  theta0={theta_0}, theta1={theta_1} learning_rate={learning_rate}.png")

theta_0 = 5
theta_1 = 10
train()

for theta_0 in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    for theta_1 in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        train()
