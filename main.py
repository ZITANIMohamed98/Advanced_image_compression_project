#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lossy digital image compression.

@author: khe
"""
import rawpy
import cv2
from PIL import Image
import numpy as np
from multiprocessing.pool import Pool
import utils
import os

###############################################################################
# Instantiation
###############################################################################
lum_downsample = utils.Downsampling(ratio='4:4:4')
chr_downsample = utils.Downsampling(ratio='4:2:0')
image_block = utils.ImageBlock(block_height=8, block_width=8)
dct2d = utils.DCT2D(norm='ortho')
quantization = utils.Quantization()
zigzagScanning = utils.ZigzagScanning()
rle = utils.RLE()
entropy = utils.Entropy()
###############################################################################
# Preprocess
###############################################################################
# Read raw image file as array
raw = rawpy.imread(os.path.join('images', 'DSC05719.ARW'))

# Postprocess image array (Bayer filter -> RGB)
rgb_img = raw.postprocess()

# Colorspace transform (RGB -> YCrCb)
ycc_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)

# Center
ycc_img = ycc_img.astype(int)-128

# Downsampling
Y = lum_downsample(ycc_img[:,:,0])
Cr = chr_downsample(ycc_img[:,:,1])
Cb = chr_downsample(ycc_img[:,:,2])
ycc_img = np.stack((Y, Cr, Cb), axis=2)

# Create 8x8 blocks
blocks, indices = image_block.forward(ycc_img)

###############################################################################
# Compression
###############################################################################

def process_block(block, index, toEncode_Y, toEncode_CbCr):

    #Prediction  -> Prediction error
    
    # DCT
    encoded = dct2d.forward(block)
    if index[2] == 0:
        channel_type = 'lum'
    else:
        channel_type = 'chr'
        
    # Quantization
    encoded = quantization.forward(encoded, channel_type)
    
    # RLE + zigzag scanning
    encoded = zigzagScanning.forward(encoded)
    encoded = rle.forward(encoded)
    
    assert channel_type in ('lum', 'chr')
        
    if channel_type == 'lum':
        toEncode_Y.append(encoded)
    else:
        toEncode_CbCr.append(encoded)
    
    

    
def reconstruct_blocks(decodedY, decodedCbCr,index, reconstructed_frame):
    if index[2] == 0:
        channel_type = 'lum'
    else:
        channel_type = 'chr'
    
    assert channel_type in ('lum', 'chr')
        
    if channel_type == 'lum':
        decoded = decodedY[0]
        decodedY.pop(0)
    else:
        decoded = decodedCbCr[0]
        decodedCbCr.pop(0)
        
    decoded = rle.backward(decoded)
    # Reverse RLE + zigzag scanning
    decoded = zigzagScanning.backward(decoded)
    # Dequantization
    decoded = quantization.backward(decoded, channel_type)
    
    # Reverse DCT
    reconstructed_block = dct2d.backward(decoded)
    
    reconstructed_frame.append(reconstructed_block)

# Use Pool from the multiprocessing library becasue the compression task is 
# highly parallelizable. The same operation is performed on different blocks
# where there is no dependency among the data. 
#compressed = np.array(Pool().starmap(process_block, zip(blocks, indices)))
toEncode_Y = []
toEncode_CbCr = []
for i in range(0,int(blocks.size/64)-1):
  process_block(blocks[i] ,indices[i] ,toEncode_Y ,toEncode_CbCr)
  
# Huffman coding for Luma and chroma separetly
encodedY, codebookY = entropy.huffman_encoding(toEncode_Y)
encodedCbCr, codebookCbCr= entropy.huffman_encoding(toEncode_CbCr)

# calculating the compression ratio
compression_ratio = rgb_img.shape[0]*rgb_img.shape[1]*24/(sum(len(word) for word in encodedY)+sum(len(word) for word in encodedCbCr))
print(compression_ratio)

# Huffman decoding for Luma and chroma separetly
decodedY = entropy.huffman_decoding(encodedY, codebookY)
decodedCbCr = entropy.huffman_decoding(encodedCbCr, codebookCbCr)

#Reconstruct the Frame
reconstructed_frame = []
for i in range(0,int(blocks.size/64)-1):
    reconstruct_blocks(decodedY, decodedCbCr,indices[i], reconstructed_frame)
    

###############################################################################
# Postprocess
###############################################################################
# Reconstruct image from blocks
ycc_img_compressed = image_block.backward(reconstructed_frame, indices)

# Rescale
ycc_img_compressed = (ycc_img_compressed+128).astype('uint8')

# Transform back to RGB
rgb_img_compressed = cv2.cvtColor(ycc_img_compressed, cv2.COLOR_YCrCb2RGB)

# Write to file
Image.fromarray(rgb_img_compressed).save(os.path.join('images', 'result.jpeg'))
