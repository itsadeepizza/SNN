import numpy as np
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

"""
https://deepai.org/dataset/mnist
Pixels are organized row-wise. Pixel values are 0 to 255.
0 means background (white), 255 means foreground (black)

A simpler approach, each pixel has at each instant a probability to spike given 
by its gray value, in the 0-255 scale
"""

p_0 = 0.1

class SpikingDatasetSimple():
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
            img_prob = img / 255 * p_0
            # Each pixel has a probability given by img_probs to spike at each instant.
            # The process has no memory
            random_matrix = np.random.rand(self.T, self.num_of_rows, self.num_of_cols)
            # Stack self.T copies of img_prob
            img_prob = np.stack([img_prob] * self.T)

            pixel_spiking = (random_matrix < img_prob).nonzero()
            #Swap x and y
            swapped = np.swapaxes(pixel_spiking, 0, 1)
            bins = [[] for _ in range(self.T)]
            for t, x, y in swapped:
                bins[t].append((x, y))

            return bins, label

    def __len__(self):
        return self.num_of_imgs


if __name__ == "__main__":
    spiking_dataset = SpikingDatasetSimple(("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte"))
    for image, label in tqdm(spiking_dataset):
        print(image, label)





