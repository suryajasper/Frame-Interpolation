from video_manip import *
import argparse
import os
import os.path
import cv2
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--orig", type=str, required=True, help='path to original video file')
parser.add_argument("--interp", type=str, required=True, help='path to interpolated video file')
parser.add_argument("--debug", type=bool, default=False, help='print logs')
args = parser.parse_args()

framesOrig = getFrames(args.orig)
if args.debug:
	print(len(framesOrig), '*', np.shape(framesOrig[0]))
	print(framesOrig[0][0][0][0], ',',framesOrig[0][0][0][1], ',',framesOrig[0][0][0][2])

framesInterp = getFrames(args.interp)
if args.debug:
	print(len(framesInterp), '*', np.shape(framesInterp[0]))
	print(framesInterp[0][0][0][0], ',',framesInterp[0][0][0][1], ',',framesInterp[0][0][0][2])

width = np.shape(framesOrig[0])[1]
height = np.shape(framesOrig[0])[0]
mse = 0

goThru = min(len(framesOrig), len(framesInterp))-1

pbar = tqdm(total=goThru)

for frame in range(goThru):
	pbar.update(1)
	for color in range(3):
		for x in range(width):
			for y in range(height):
				mse += (framesOrig[frame][y][x][color]-framesInterp[frame][y][x][color])/256/(framesOrig[frame][y][x][color]+1)

pbar.close()

mse /= width*height*3*goThru

print('{}%'.format(100-100*mse))

