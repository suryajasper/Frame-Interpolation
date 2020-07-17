# Cartoon Convolutional Neural Network

Inspired by unet architecture. Rewritten to output full images.
Because we have two convolutional stream, in deconvolution we combine two
skip connections instead of one.

For unet implementation we use:
https://github.com/milesial/Pytorch-UNet