import numpy as np
import cv2
import os
from os.path import isfile, join

def getFrames(url):
	frames = []
	vidcap = cv2.VideoCapture(url)
	success,image = vidcap.read()
	count = 0
	while success:
		success,image = vidcap.read()
		frames.append(image)
	return frames

def remove_every_other(my_list):
	return my_list[::4]
	pass

def getFramesAlternate(url):
	frames = []
	vidcap = cv2.VideoCapture(url)
	success,image = vidcap.read()
	count = 0
	while success:
		success,image = vidcap.read()
		frames.append(image)
	return remove_every_other(frames)

def framesToVideo(frames, fps, pathOut):
	height, width, layers = frames[0].shape
	size = (width, height)

	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	for i in range(len(frames)):
		# writing to a image array
		out.write(frames[i])
	out.release()

