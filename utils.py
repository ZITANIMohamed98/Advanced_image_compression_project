#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image compression util functions.

@author: khe
"""
import numpy as np
from scipy.fft import dct
from scipy.signal import convolve2d

class Downsampling():
    def __init__(self, ratio='4:2:0'):
        assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
        self.ratio = ratio
        
    def __call__(self, x):
        # No subsampling
        if self.ratio == '4:4:4':
            return x
        else:
            # Downsample with a window of 2 in the horizontal direction
            if self.ratio == '4:2:2':
                kernel = np.array([[0.5], [0.5]])
                out = np.repeat(convolve2d(x, kernel, mode='valid')[::2,:], 2, axis=0)
            # Downsample with a window of 2 in both directions
            else:
                kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
                out = np.repeat(np.repeat(convolve2d(x, kernel, mode='valid')[::2,::2], 2, axis=0), 2, axis=1)
            return np.round(out).astype('int')

class ImageBlock():
    def __init__(self, block_height=8, block_width=8):
        self.block_height = block_height
        self.block_width = block_width
        self.left_padding = self.right_padding = self.top_padding = self.bottom_padding = 0
    
    def forward(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_channel = image.shape[2]
    
        # Vertical padding
        if self.image_height % self.block_height != 0:
            vpad = self.image_height % self.block_height
            self.top_padding = vpad // 2 
            self.bottom_padding = vpad - self.top_padding
            image = np.concatenate((np.repeat(image[:1], self.top_padding, 0), image, 
                                    np.repeat(image[-1:], self.bottom_padding, 0)), axis=0)
            
        # Horizontal padding
        if self.image_width % self.block_width != 0:
            hpad = self.image_width % self.block_width
            self.left_padding = hpad // 2 
            self.right_padding = hpad - self.left_padding
            image = np.concatenate((np.repeat(image[:,:1], self.left_padding, 1), image, 
                                    np.repeat(image[:,-1:], self.right_padding, 1)), axis=1)
    
        # Update dimension
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

        # Create blocks
        blocks = []
        indices = []
        for i in range(0, self.image_height, self.block_height):
            for j in range(0, self.image_width, self.block_width):
                for k in range(self.image_channel):
                    blocks.append(image[i:i+self.block_height, j:j+self.block_width, k])
                    indices.append((i,j,k))
                    
        blocks = np.array(blocks)
        indices = np.array(indices)
        return blocks, indices
    
    def backward(self, blocks, indices):
        # Empty image array
        image = np.zeros((self.image_height, self.image_width, self.image_channel)).astype(int)
        for block, index in zip(blocks, indices):
            i, j, k = index
            image[i:i+self.block_height, j:j+self.block_width, k] = block
            
        # Remove padding
        if self.top_padding > 0:
            image = image[self.top_padding:,:,:]
        if self.bottom_padding > 0:
            image = image[:-self.bottom_padding,:,:] 
        if self.left_padding > 0:
            image = image[:,self.left_padding:,:]
        if self.right_padding > 0:
            image = image[:,:-self.right_padding,:]
        return image

class DCT2D():
    def __init__(self, norm='ortho'):
        if norm is not None:
            assert norm == 'ortho', "norm needs to be in {None, 'ortho'}"
        self.norm = norm
    
    def forward(self, x):
        out = dct(dct(x, norm=self.norm, axis=0), norm=self.norm, axis=1)
        return out
    
    def backward(self,x):
        out = dct(dct(x, type=3, norm=self.norm, axis=0), type=3, norm=self.norm, axis=1)
        return np.round(out)

class ZigzagScanning():
    
    def forward(self, block):
        """Perform zig-zag scanning on an 8x8 matrix."""
        scan = []
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:  # even diagonal
                    scan.append(block[i][j])
                else:  # odd diagonal
                    scan.append(block[j][i])
        return scan

    def backward(self,scan):
        """Perform reverse zig-zag scanning on a list of 64 elements."""
        matrix = [[0 for _ in range(8)] for _ in range(8)]
        idx = 0
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:  # even diagonal
                    matrix[i][j] = scan[idx]
                else:  # odd diagonal
                    matrix[j][i] = scan[idx]
                idx += 1
        return matrix
class RLE():
    def forward(self,seq):
        encoded_seq = []
        current_value = seq[0]
        count = 1

        for i in range(1, len(seq)):
            if seq[i] == current_value:
                count += 1
            else:
                encoded_seq.append((current_value, count))
                current_value = seq[i]
                count = 1

        encoded_seq.append((current_value, count))

        return encoded_seq
    
    def backward(self,encoded_seq):
        decoded_seq = []
        for value, count in encoded_seq:
            decoded_seq.extend([value] * count)
        return decoded_seq
    
class Entropy():
    def forward(self,seq):
        # create a dictionary to store the frequency of each symbol
        freq = {}
        for symbol in seq:
            if symbol in freq:
                freq[symbol] += 1
            else:
                freq[symbol] = 1

        # calculate the probability of each symbol
        total = sum(freq.values())
        prob = {symbol: freq[symbol] / total for symbol in freq}

        # initialize the range and the current value
        low = 0
        high = 1
        value = 0

        # encode each symbol using arithmetic coding
        for symbol in seq:
            range_size = high - low
            high = low + range_size * sum(prob[s] for s in prob if s <= symbol)
            low = low + range_size * sum(prob[s] for s in prob if s < symbol)
            value = (value * total) + freq[symbol] * sum(prob[s] for s in prob if s < symbol)

        # return the encoded value and the probability dictionary
        return int(value), prob

    def backward(self, encoded_value, freq):
    # check if the freq dictionary is empty
        if not freq:
            return []

        # create a dictionary to store the frequency of each symbol
        total = sum(freq.values())

        # initialize the range and the current value
        low = 0
        high = 1
        value = encoded_value

        # initialize the decoded sequence
        decoded_seq = []

        # decode each symbol using arithmetic coding
        for i in range(len(freq)):
            # calculate the range size and the symbol interval
            range_size = high - low
            symbol_interval = (value - low) / range_size

            # find the symbol that corresponds to the symbol interval
            symbol = None
            cum_prob = 0
            for symbol, prob in freq.items():
                cum_prob += prob / total
                if cum_prob > symbol_interval:
                    break

            # update the range and the current value for the next symbol
            high = low + range_size * cum_prob
            low = low + range_size * (cum_prob - prob / total)
            value = (value - low) / range_size * total

            # add the decoded symbol to the sequence
            decoded_seq.append(symbol)

        return decoded_seq

class Quantization():
    # Quantiztion matrices
    # https://www.impulseadventure.com/photo/jpeg-quantization.html
    
    # Luminance
    Q_lum = np.array([[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]])
    # Chrominance
    Q_chr = np.array([[17,18,24,47,99,99,99,99],
                      [18,21,26,66,99,99,99,99],
                      [24,26,56,99,99,99,99,99],
                      [47,66,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99]])
    
    def forward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')
        
        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = np.round(x/Q)
        return out
    
    def backward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')
        
        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = x*Q
        return out
