import numpy as np
import pandas as pd
from tqdm import tqdm
from mnist_encoder3 import SpikingDataset, SpikingImageEncoder
from matplotlib import pyplot as plt
import logging
import sys


# ╔╦╗╔═╗╔╦╗╔═╗╦
# ║║║║ ║ ║║║╣ ║
# ╩ ╩╚═╝═╩╝╚═╝╩═╝


class SNN:
    pass


class SpikingLayer(SNN):
    """
    Spiking layer for BP-STDP (https://arxiv.org/pdf/1711.04214.pdf)
    """
    def __init__(self, prev_layer:SNN, output_dim:int, theta=0.9, learning_rate=0.005):
        self.prev_layer = prev_layer
        self.input_dim = self.prev_layer.output_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
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
        self.W += self.learning_rate * delta_W
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


# threshold = 0.01  # firing threshold for gaussians
T = 100  # Number of time steps

hidden_dim = 100
output_dim = 10
learning_rate = 0.005
epsilon = 4
theta_0 = 0.9
theta_1 = 5

# Generate dataset
spiking_dataset = SpikingDataset(("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte"), T=T)


def train_model(dataset, layers):
    subset = np.random.choice(dataset, 1000, replace=False)
    encoder_layer, hidden_layer, output_layer = layers
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

            if t % epsilon == 0:
                # Update weights
                csi = output_layer.get_csi(label)
                reward = output_layer.backward(csi)
                reward = hidden_layer.backward(reward)
    return encoder_layer, hidden_layer, output_layer


def eval_model(dataset, layers):
    subset = np.random.choice(dataset, 1000, replace=False)
    encoder_layer, hidden_layer, output_layer = layers
    correct_predictions = 0
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

        # Check if prediction is correct
        pred = np.argmax(total_output)
        if pred == target:
            correct_predictions += 1
    accuracy = correct_predictions / len(subset)
    return accuracy


import optuna
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 10, 150, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    epsilon = trial.suggest_int("epsilon", 1, 10)
    theta_0 = trial.suggest_float("theta_0", 0.1, 10, log=True)
    theta_1 = trial.suggest_float("theta_1", 0.1, 10, log=True)

    # Define model
    # ╦ ╔╗╔ ╦ ╔╦╗
    # ║ ║║║ ║  ║
    # ╩ ╝╚╝ ╩  ╩
    # Initialize output and hidden layer
    encoder_layer = SpikingImageEncoder(28, 28, T, epsilon=epsilon)
    hidden_layer = SpikingLayer(encoder_layer, hidden_dim, theta=theta_0, learning_rate=learning_rate)
    output_layer = SpikingLayer(hidden_layer, output_dim, theta=theta_1, learning_rate=learning_rate)

    # ╔╦╗╦═╗╔═╗╦╔╗╔
    #  ║ ╠╦╝╠═╣║║║║
    #  ╩ ╩╚═╩ ╩╩╝╚╝

    all_rows = list(range(len(spiking_dataset)))
    train = np.random.choice(all_rows, int(0.7 * len(all_rows)), replace=False)
    test = np.array([i for i in all_rows if i not in train])

    # Train the model
    layers = (encoder_layer, hidden_layer, output_layer)
    for epoch in range(10):
        layers = train_model(train, layers)

    # Eval the model
    accuracy = eval_model(test, layers)
    return accuracy



# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "snn"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)


study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")

study.optimize(objective, n_trials=30)

# Run the dashboard with the command below:
# optuna-dashboard sqlite:///snn.db