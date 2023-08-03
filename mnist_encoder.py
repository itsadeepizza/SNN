from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d

np.seterr(divide='ignore', invalid='ignore')

"""
https://deepai.org/dataset/mnist
Pixels are organized row-wise. Pixel values are 0 to 255.
0 means background (white), 255 means foreground (black)
"""

def visual_print(data_element):
    img = data_element[0][0]
    label = data_element[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] == 0):
                print(" ", end="")
            else:
                print("@", end="")
        print()
    print(label)

def save_img(array, path):

    # (-inf,+inf) to [0,255]
    min_el = np.min(array)
    shifted = array - min_el if min_el < 0 else array
    scaled = 255 * (shifted / np.max(shifted)) if np.max(shifted) > 0 else shifted

    img = Image.fromarray(scaled.astype(np.uint8))
    img.save(path)

def conv2d(array, kernel): # assuming: same input's dim for output, stride 1
    return convolve2d(array, kernel, mode='same')

    # result = numpy.zeros(array.shape)
    #
    # pad_size = int(numpy.floor(kernel.shape[0]/2))
    # padded = numpy.pad(array, pad_size) # pad with 0s for dimensions
    #
    # for i in range(0, padded.shape[0]-kernel.shape[0]+1):
    #     for j in range(0, padded.shape[1]-kernel.shape[1]+1):
    #         sliding_window = padded[i: i+kernel.shape[0], j: j+kernel.shape[1]]
    #         result[i][j] = numpy.sum( sliding_window * kernel )
    #
    # return result

def DoG_kernel(sigma1, sigma2 ,N = 7, M = 7):

    assert(N%2 != 0)
    assert(M%2 != 0)

    def gaussian(i,j, sigma):
        return ( \
                    np.exp( \
                        - (np.power(i, 2) + np.power(j, 2)) / (2 * np.power(sigma, 2)) \
                ) \
                    / (2 * np.pi * np.power(sigma, 2)) \
               )

    kernel = np.zeros([N, M])

    for i in range(N):
        for j in range(M):
            _i = i - np.floor(N / 2)
            _j = j - np.floor(M / 2)
            kernel[i][j] = gaussian(_i, _j ,sigma1) - gaussian(_i, _j ,sigma2)

    return kernel

def img_to_spikes(np_img, threshold = 50): # 50 works bad, 10 more like in paper



    conv_on = conv2d(np_img, DoG_kernel(sigma1 = 1, sigma2 = 2))
    conv_off = conv2d(np_img, DoG_kernel(sigma1 = 2, sigma2 = 1))

    mask_on = np.ma.masked_greater(conv_on, threshold).mask
    spikes_on = 255 * mask_on.astype(np.float64) \
               if mask_on.shape == np_img.shape    \
               else np.zeros(np_img.shape)

    mask_off = np.ma.masked_greater(conv_off, threshold).mask
    spikes_off = 255 * mask_off.astype(np.float64) \
               if mask_off.shape ==  np_img.shape    \
               else np.zeros(np_img.shape)

    timings_on = np.where(mask_on, 1 / conv_on, 0)
    timings_off = np.where(mask_off, 1 / conv_off, 0)

    on = np.stack((spikes_on, timings_on))
    off = np.stack((spikes_off, timings_off))
    result = np.stack((on, off)) # shape: (on/off, spike/t, xvalue, yvalue)

    return result


class SpikingImageEncoder:
    def __init__(self, W, H, T, epsilon):
        """
        :param W: width
        :param H: height
        :param bins: number of bins for each pixel
        """
        self.W = W
        self.H = H
        self.T = T
        self.output_dim = W * H
        # Last spikes
        self.recorded_spikes = np.zeros((epsilon, self.output_dim))

    def load_bins(self, bins):
        self.bins = bins
        assert (self.T == len(bins))
        self.recorded_spikes.fill(0)


    def get_state(self, t):
        """
        :param t: timestep
        :return: list of 0 or 1 if corresponding neuron is firing
        """
        out = np.zeros([self.W , self.H])
        current_bin = self.bins[t]
        out[current_bin] = 1
        spikes = out.flatten()
        self.recorded_spikes = np.roll(self.recorded_spikes, -1, axis=0)
        self.recorded_spikes[-1, :] = spikes
        return spikes



    def __call__(self, img):
        return img_to_spikes(img, threshold = 50)

class SpikingDataset():
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
            continuous_encoding = img_to_spikes(img)[0, 1, :, :]
            discrete_encoding = self.gen_bins_2d(continuous_encoding, T=self.T, W=self.W, H=self.H)
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
        # greater_0idx = 0
        # # find first non zero spike
        # # TODO: redo using numpy
        # while greater_0idx < len(flat_t):
        #     if (flat_t[sorted_indexes[greater_0idx]] == 0):
        #         greater_0idx += 1
        #     else:
        #         break
        if greater_0idx == len(flat_t):
            return np.array([np.zeros(timings.shape)] * T)

        sorted_indexes = sorted_indexes[greater_0idx:]  # remove non spiking neurons
        # convert flattened indexes to 2d indexes
        sorted_indexes_2d = np.dstack(np.unravel_index(sorted_indexes, timings.shape))[0]

        # bins_indexes = list(range(0, len(sorted_indexes_2d), int(numpy.ceil(len(sorted_indexes_2d) / T))))
        # bins_indexes.append(len(sorted_indexes_2d))
        # bins_indexes = bins_indexes[1:]

        # Build T bins for splitting neurons
        bins_indexes = np.linspace(0, len(sorted_indexes_2d), T + 1, dtype=int)[1:]
        # Split neurons into bins
        prev = 0
        for i, element in enumerate(bins_indexes):
            spiking_neurons = sorted_indexes_2d[prev:element]
            bins.append(spiking_neurons)
            prev = element

        return bins
        # return SpikingImageEncoder(W, H, bins)

        # # TODO implement and use sparse matrix for this
        # for spikes in bins:
        #     bin_img = numpy.zeros(timings.shape)
        #     for i in range(len(bin_img)):
        #         for j in range(len(bin_img[0])):
        #             for neuron in spikes:
        #                 if numpy.array_equal([i, j], neuron):
        #                     # bin_img[i][j] = 255
        #                     bin_img[i][j] = 1
        #     bin_imgs.append(bin_img)

        # bins2d = numpy.array(bin_imgs)
        # return bins2d



if __name__ == "__main__":
    spiking_dataset = SpikingDataset(("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte"))
    for image_encoded, label in tqdm(spiking_dataset):
        pass


    #save_img(conv2d(image,kernel1), 'conv1.png')
    #save_img(kernel2, 'k2.png')

    #dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    #dataiter = iter(dataloader)

    #data = dataiter.next()
    #print("label shape ",data[1].shape)
    #print("label",data[1])




