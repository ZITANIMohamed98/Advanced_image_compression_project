#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image compression util functions.

@author: khe
"""
import numpy as np
from scipy.fft import dct
from scipy.signal import convolve2d
from heapq import heappush, heappop, heapify
from collections import defaultdict

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
    def huffman_encoding(self,seq):
        if not seq:
            return "", None
        freq = defaultdict(int)
        for lst in seq:
            for symbol in lst:
                freq[symbol] += 1

# Step 2: Create a binary tree
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Step 3: Traverse the tree to assign unique binary codes to each symbol
        huff = dict(heappop(heap)[1:])
    # Step 4: Encode the input sequence using the binary codes
        encoded_seq = ""
        for lst in seq:
            for symbol in lst:
                encoded_seq += huff[symbol]
                
        return encoded_seq, huff       
    def huffman_decoding(encoded_seq, huff):
        decoded_seq = ""
        temp = ""
        for bit in encoded_seq:
            temp += bit
            for symbol, code in huff.items():
                if code == temp:
                    decoded_seq += str(symbol)
                    temp = ""
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
