import numpy as np
import cv2
import os
from os.path import isfile, join

def getFrames(url):
	frames = []
	vidcap = cv2.VideoCapture(url)
	success,image = vidcap.read()
	while success:
		success,image = vidcap.read()
		frames.append(image)
	return frames

def framesToVideo(frames, fps, out):
	height, width, layers = frames[0].shape
	size = (width, height)
	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	for i in range(len(frame_array)):
		out.write(frame_array[i])
	out.release()

g = getFrames('../datasets/60fps/0.mp4')
framesToVideo(g, 30, 'stu.mp4')
