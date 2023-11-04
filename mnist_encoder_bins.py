from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d

np.seterr(divide='ignore', invalid='ignore')

"""
https://deepai.org/dataset/mnist
Pixels are organized row-wise. Pixel values are 0 to 255.
0 means background (white), 255 means foreground (black)

Each pixel spike max once at a given instant. The instant is given
by it's relative gray intensity, compared to other pixels
"""

class SpikingDatasetBins():
    W = 27
    H = 27
    def __init__(self, filenames, T=10):
        """
        :param filenames: [train, label]
        :param T: even number of timesteps
        """
        assert T%2 == 0 # even number of timesteps
        self.T = T
        try:
            self.f_train = open(filenames[0], mode='rb')
            self.f_label = open(filenames[1], mode='rb')
        except:
            print("cant open files :C")
            quit()

        self.f_train.read(4) # magic number
        self.f_label.read(4) # magic number

        self.num_of_imgs = int.from_bytes(self.f_train.read(4), "big")
        self.num_of_rows = int.from_bytes(self.f_train.read(4), "big")
        self.num_of_cols = int.from_bytes(self.f_train.read(4), "big")

        self.offset_train = 16
        self.offset_label = 8

        print(f"dataset created: {self.num_of_imgs} {self.num_of_rows}x{self.num_of_cols} images found")

    def __getitem__(self,index):
        if (index < self.num_of_imgs):
            self.f_train.seek(self.offset_train + index * self.num_of_rows * self.num_of_cols)
            self.f_label.seek(self.offset_label + index)

            label_ind = int.from_bytes(self.f_label.read(1), "big")
            label = np.zeros((10))
            label[label_ind] = 1

            img = []
            for _ in range(self.num_of_rows):
                row = []
                for __ in range(self.num_of_cols):
                    pixel = int.from_bytes(self.f_train.read(1), "big")
                    row.append(pixel)
                img.append(row)

            img = np.array(img)
            img = img[1:,1:] # slicing to have 27x27 pics as in the paper
            # continuous_encoding = img_to_spikes(img)[0, 1, :, :]
            discrete_encoding = self.gen_bins_2d(img / 255, T=self.T, W=self.W, H=self.H)
            return (discrete_encoding,label)

        else:
            raise ValueError("index error")

    def __len__(self):
        return self.num_of_imgs

    @staticmethod
    def gen_bins_2d(timings, T=10, W=27, H=27):
        """
        :param timings: a 2d array nxn of timings, where n is the number of neurons
        :param T: an even number of bins
        :return: T x n x n array, values t, x, y says if neuron x,y spiked in time t or not
        """
        bin_imgs = []
        bins = []
        flat_t = timings.flatten()

        sorted_indexes = flat_t.argsort()
        greater_0idx = (timings==0).sum()
        if greater_0idx == len(flat_t):
            return np.array([np.zeros(timings.shape)] * T)

        sorted_indexes = sorted_indexes[greater_0idx:]  # remove non spiking neurons
        # convert flattened indexes to 2d indexes
        sorted_indexes_2d = np.dstack(np.unravel_index(sorted_indexes, timings.shape))[0]

        # Build T bins for splitting neurons
        bins_indexes = np.linspace(0, len(sorted_indexes_2d), T + 1, dtype=int)[1:]
        # Split neurons into bins
        prev = 0
        for i, element in enumerate(bins_indexes):
            spiking_neurons = sorted_indexes_2d[prev:element]
            bins.append(spiking_neurons)
            prev = element

        return bins

if __name__ == "__main__":
    spiking_dataset = SpikingDatasetBins(("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte"))
    for image_encoded, label in tqdm(spiking_dataset):
        pass




