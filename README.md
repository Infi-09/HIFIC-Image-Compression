# High Fidelity Generative Compression

Pytorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/). This repo also provides general utilities for lossless compression that interface with Pytorch. For the official (TensorFlow) code release, see the [TensorFlow compression
repo](https://github.com/tensorflow/compression/tree/master/models/hific).

## About

This repository defines a model for learnable image compression based on the paper ["High-Fidelity Generative Image Compression" (HIFIC) by Mentzer et. al.](https://hific.github.io/). The model is capable of compressing images of arbitrary spatial dimension and resolution up to two orders of magnitude in size, while maintaining perceptually similar reconstructions. Outputs tend to be more visually pleasing than standard image codecs operating at higher bitrates.

