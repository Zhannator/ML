# Public libraries
import cv2
import matplotlib.pyplot as plt

def main():
	# Iterate through facial images
	for i in range(1, 41, 1):
		for j in range(1, 11, 1):
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm")
			rows, columns = img.shape
			
if __name__ == "__main__":
	main()