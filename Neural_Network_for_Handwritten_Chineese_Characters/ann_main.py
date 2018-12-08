# Public libraries
import sys
import cv2
import numpy as np
import math
import random
import itertools

def activation_function():
	print "TODO"

def ann():
	print #TODO
	
	# Input layer (images)
	
	# Hidden layer (100 neurons)
	
	# Output layer (62 neurons)

def read_data():
	img_height = 900
	img_width = 1200
	if (len(sys.argv) > 1):
		if ("-resize" in sys.argv):
			img_height = 90 # 225
			img_width = 120 # 300
	num_of_pixels = img_height * img_width
	num_of_classes = 62
	num_of_images_per_class = 55
	num_of_images = num_of_classes * num_of_images_per_class
	num_of_testing_images_per_class = int(num_of_images_per_class * 0.1) # 10 % testing images
	num_of_training_images_per_class = num_of_images_per_class - num_of_testing_images_per_class # 90 % testing images
	num_of_testing_images = num_of_classes * num_of_testing_images_per_class # 10 % testing images
	num_of_training_images = num_of_classes * num_of_training_images_per_class # 90 % testing images
	images_training = np.zeros((num_of_training_images, num_of_pixels)) # each img as row vector
	images_testing = np.zeros((num_of_testing_images, num_of_pixels)) # each img as row vector
	classes_training = np.zeros(num_of_training_images) # classes
	classes_testing = np.zeros(num_of_testing_images) # classes

	if (len(sys.argv) > 1):
		if ("-resize" in sys.argv):
			print "\nAll of the images will be resized form (900x1200) to ({}x{})...\n".format(img_height, img_width)
	
	counter_training = 0
	counter_testing = 0
	for i in range(1, num_of_classes + 1, 1):
		i_padded = str(i).zfill(3)
		# Each class has 55 samples. In each class, randomly select 5 samples as testing data and the rest 50 samples as training data
		rand_indexes = random.sample(range(1, num_of_images_per_class + 1), num_of_images_per_class)
		rand_indexes_len = len(rand_indexes)
		# Training images
		for j in range(num_of_training_images_per_class):
			j_padded = str(rand_indexes[j]).zfill(3)
			# Read image
			img = cv2.imread("data/Sample" + i_padded + "/img" + i_padded + "-" + str(j_padded) + ".png", 0) * (1.0 / 255.0) 
			if (len(sys.argv) > 1):
				if ("-resize" in sys.argv):
					img = cv2.resize(img, (img_height, img_width)) 
			# 2D image matrix to 1D row vector
			images_training[counter_training] = (np.array(img)).flatten()
			classes_training[counter_training] = i
			counter_training = counter_training + 1
		# Testing images
		for j in range(num_of_training_images_per_class, rand_indexes_len, 1):
			j_padded = str(rand_indexes[j]).zfill(3)
			# Read image
			img = cv2.imread("data/Sample" + i_padded + "/img" + i_padded + "-" + str(j_padded) + ".png", 0) * (1.0 / 255.0) 
			if (len(sys.argv) > 1):
				if ("-resize" in sys.argv):
					img = cv2.resize(img, (img_height, img_width)) 
			# 2D image matrix to 1D row vector
			images_testing[counter_testing] = (np.array(img)).flatten()
			classes_testing[counter_testing] = i
			counter_testing = counter_testing + 1
	# Convert images to type double
	images_training = images_training.astype(np.double)
	images_testing = images_testing.astype(np.double)
	# Transpose image matrixes: row images -> column images
	#images_training = np.transpose(images_training)
	#images_testing = np.transpose(images_testing)
	
	return images_training, classes_training, images_testing, classes_testing

################################################################################
# Main pca function
################################################################################
def pca(img_training, img_testing):
	# Feature extraction using PCA
	U, m = pca_analysis(np.transpose(img_training))
	W_training = pca_extract_features(U, img_training, m)	
	W_testing = pca_extract_features(U, img_testing, m)

	# Normalize data
	W_training = pca_normalize(W_training)	
	W_testing = pca_normalize(W_testing)
	
	return W_training, W_testing

################################################################################
# Calculate eigenfaces and their according eigenvalues
################################################################################
def pca_analysis(T):
	# Calculate mean face
	m = (T.mean(1))[0]
	
	# Subtract mean from each column vector in (a.k.a. center images)
	A = T - m
	
	#print "\nA Shape: {}\n".format(A.shape)
	
	# Calculate convergence matrix
	AtA = np.matmul(np.transpose(A), A)
	
	#print "\nAtA Shape: {}\n".format(AtA.shape)
	
	# Calculate eigenvalues and eigenfaces of AtA
	eigenvalues, eigenfaces = np.linalg.eig(AtA)
	
	# Sort eigenvalues and eigenvectors (largest to smallest)
	idx = eigenvalues.argsort()[::-1]   
	eigenvalues = eigenvalues[idx]
	eigenfaces = eigenfaces[:,idx]	

	# Decide how many eigenfaces are enough to represent variance in our training set - at least 95 % variance
	k = reduce_number_of_eigenvectors(eigenvalues, 0.95)

	# Dominant eigenvectors
	V = eigenfaces[:, 0 : k]
	
	# Calculate U (most important eigenvectors from AAt) by multiplying A and V (only most important eigenvectors)
	U = np.matmul(A, V)

	return U, m

################################################################################
# Extract and return features from each image in images (a.k.a. Projection)
################################################################################
def pca_extract_features(U, images, m):
	U_T = np.transpose(U)
	W_training = []
	num_images = len(images)
	images_unique = images - m
	for i in range(num_images):
		W_training.append(np.dot(U_T, images_unique[i]))
	return np.array(W_training)

################################################################################
# Normalize data using mean and standard deviation
################################################################################	
def pca_normalize(data):
	# Column-wise subtract the mean and divide by the std deviation
	rows, columns = data.shape
	for r in range(rows):
		data[r] = (data[r] - (data[r]).mean(0)) / np.std(data[r])
		
	return data
	
################################################################################
# Main lda function
################################################################################
def lda(img_training, classes_training, img_testing, num_of_classes, num_of_images_per_class, z):	
	# Step 1 - calculate separability between different classes (distance between means of different classes)
	# Between-Class Matrix (Sb)
	# Step 2 - calculate distance between means and the samples of each class
	# Within-Class Matrix (Sw)
	z = img_training.shape[1]
	Sb, Sw = lda_sb_and_sw(img_training, classes_training, num_of_classes, num_of_images_per_class, z)
	
	# Step 3 - Construct the lower dimensional space (Vk) that 
	# Maximizes Between-Class Matrix and minimizes Within-Class Matrix
	Swt = np.transpose(Sw)
	print "\nWorking on np.dot for SwtSb...\n"
	SwtSb = np.dot(Swt, Sb)
	print "\nWorking on np.linalg.eigh...\n"
	eigenvalues, eigenvectors = np.linalg.eigh(SwtSb)
		
	# Decide how many eigenvectors are enough to represent variance in our training set - at least 95 % variance
	k = reduce_number_of_eigenvectors(eigenvalues, 0.95)
	# Dominant eigenvectors
	Vk = eigenvectors[:, 0 : k]
	Vk = np.transpose(Vk)
	
	# Step 4 - Project our original data into lower-dimensional space
	Vk_training = np.dot(Vk, np.transpose(img_training))
	Vk_testing = np.dot(Vk, np.transpose(img_testing))
	
	return np.transpose(Vk_training), np.transpose(Vk_testing)

################################################################################
# Lda helper, computes Between-Class Matrix (Sb) and Within-Class Matrix (Sw)
################################################################################
def lda_sb_and_sw(img_training, classes_training, num_of_classes, num_of_images_per_class, z):
	# Compute mean of each class
	img_rows, img_columns = img_training.shape
	m = np.zeros((num_of_classes, img_columns))
	classes_training = classes_training.astype(np.int)
	for r in range(img_rows):
		m[classes_training[r] - 1] = m[classes_training[r] - 1] + img_training[r]
	for r in range(num_of_classes):
		m[r] = m[r] / num_of_images_per_class
	
	# Compute total mean of all data
	m_total = (img_training.mean(0))[0]
	
	# Calculate separability between different classes (distance between means of different classes)
	# "Between-Class Matrix" (Sb)
	Sb = np.zeros((z, z))
	n = 8 # constant - number of training images per class
	print "\nWorking on np.outer fro Sb...\n"
	for r in range(num_of_classes):
		class_mean_minus_total = m[r] - m_total
		outer_product = n * np.outer(class_mean_minus_total, class_mean_minus_total)
		Sb = Sb + outer_product * float(5)
	
	# Calculate distance between means and the samples of each class
	# "Within-Class Matrix" (Sw)	
	Sw = np.zeros((z, z))
	print "\nWorking on np.outer fro Sw...\n"
	for r in range(num_of_classes):
		for img_r in range(img_rows): 
			img_minus_class_mean = img_training[img_r] - m[classes_training[img_r] - 1]
			outer_product = np.outer(img_minus_class_mean, img_minus_class_mean)
			Sw = Sw + outer_product * float(5)
			
	return Sb, Sw

################################################################################
# Calculate minimum number of eigenvectors needed to capture min_variance
################################################################################
def reduce_number_of_eigenvectors(eigenvalues_training, min_variance):
	eigenvalues_training_len = len(eigenvalues_training)
	eigenvalues_training_sum = np.sum(eigenvalues_training)
	for k in range(eigenvalues_training_len):
		v = np.sum(eigenvalues_training[0:k]) / eigenvalues_training_sum
		if v >= min_variance:
			return k + 1 # Add one because k count starts at 0
			
def main():	
	# Constants
	num_of_classes = 62
	num_of_images_per_class = 55
	
	# Read in data - each column as an image
	print "\nCreate Training and testing data..."
	img_training, classes_training, img_testing, classes_testing = read_data()
	print "\nimg_training shape: {}\n".format(img_training.shape)
	print "\nimg_testing shape: {}\n".format(img_testing.shape)
	
	# PCA
	if (len(sys.argv) > 1):
		if ("-pca" in sys.argv):
			print "\nUsing PCA...\n"
			# Feature extraction using pca
			img_training, img_testing = pca(img_training, img_testing)
			print img_training.shape
			print img_testing.shape
	
	# LDA
	if (len(sys.argv) > 1):
		if ("-lda" in sys.argv):
			print "\nUsing LDA...\n"
			z = img_training.shape[1]
			img_training, img_testing = lda(img_training, classes_training, img_testing, num_of_classes, num_of_images_per_class, z)
			print img_training.shape
			print img_testing.shape
	
	#ANN
	
	# Result of ANN is a matrix of 1*62
	
if __name__ == "__main__":
	main()