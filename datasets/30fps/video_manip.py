import numpy as np
import cv2
import os
from os.path import isfile, join

def getFrames(url):
	frames = []
	vidcap = cv2.VideoCapture('0.mp4')
	success,image = vidcap.read()
	count = 0
	while success:
		success,image = vidcap.read()
		frames.append(image)
	return frames

def framesToVideo(frames, fps, out):
	height, width, layers = frames[0].shape
	size = (width, height)

	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
    	# writing to a image array
		out.write(frame_array[i])
	out.release()