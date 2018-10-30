# Public libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers

def main():
	
	# ----------------------------------------------
	# FEATURE EXTRACTION - PCA
	# ----------------------------------------------
	
	img_height = 112
	img_width = 92
	num_of_pixes = img_height * img_width
	num_of_people = 40
	num_of_faces_per_person = 10
	num_of_faces = num_of_people * num_of_faces_per_person
	Y = [1, -1] # labels
	X = [] # data matrix
	
	# Iterate through facial images
	for i in range(1, num_of_people + 1, 1):
		for j in range(1, num_of_faces_per_person + 1, 1):
			# Read image
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm", 0)
			rows, columns = img.shape
			# 2D image matrix to 1D row vector
			img = (np.array(img)).flatten()
			X.append(img)
	
	# Calculate average / mean face
	x_mean = (np.array(X)).mean(0)
	cv2.imwrite('test.png', x_mean.reshape(img_height, img_width))
	x_mean_matrix = np.array([x_mean] * num_of_faces)
	x_mean_matrix = np.transpose(x_mean_matrix)
	# Each face image is a column vector
	X = np.transpose(X)
	# Subtract mean x from each column vector x^n in X
	F = np.subtract(X, x_mean_matrix)
	# Calculate eigenvalues and eigenvectors - if v is eigenvector of L then Av is eigenvector of S (where S = AAt)
	A = np.matmul(F, np.transpose(F))
	L = np.matmul(np.transpose(A), A)
	eigenvalues, eigenvectors = np.linalg.eig(L)

	# Solve quadratic
	alfa = cvxopt.solvers.qp(cvxopt.matrix(X), cvxopt.matrix(Y))
	print alfa

	

		
if __name__ == "__main__":
	main()