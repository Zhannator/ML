# Public libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
	
	# ----------------------------------------------
	# TRAINING
	# ----------------------------------------------
	
	# Iterate through facial images
	for i in range(1, 41, 1):
		for j in range(1, 11, 1):
			# Read image
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm", 0)
			rows, columns = img.shape
			# Detect face
			face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			faces = face_detector.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 5)
			if (len(faces) == 0):
				print "Error: no face identified in data/s" + str(i) + "/" + str(j) + ".pgm"
				return -1
			# Each image contains only 1 face
			x, y, w, h = faces[0]
			# Save face img
			cv2.imwrite("faces/s" + str(i) + "/" + str(j) + ".pgm", img[y:y+w, x:x+h])
			# Extract principal components
		return	
			
	
	# Face detection
	# Freature extraction
	# Comparison
	# Face recognition
		
if __name__ == "__main__":
	main()