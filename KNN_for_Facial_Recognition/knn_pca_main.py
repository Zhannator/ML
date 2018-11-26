# Public libraries
import sys
import cv2
import numpy as np
import math
import random
import itertools
from sklearn.neighbors import KNeighborsClassifier

################################################################################
# Main pca function
#################################################################################
def pca(img_training, img_testing):
	# Feature extraction using PCA
	U, m = pca_analysis(np.transpose(img_training))
	W_training = pca_extract_features(U, img_training, m)	
	W_testing = pca_extract_features(U, img_testing, m)
	# Normalize data
	W_training = normalize(W_training)	
	W_testing = normalize(W_testing)
	
	return W_training, W_testing

################################################################################
# Calculate eigenfaces and their according eigenvalues
#################################################################################
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
	
	# Decide how many eigenfaces are enough to represent variance in our training set - at least 95 % variance
	k = reduce_number_of_eigenvectors(eigenvalues, 0.95)

	# Dominant eigenvectors
	V = eigenfaces[:, 0 : k]
	# Calculate U (most important eigenvectors from AAt) by multiplying A and V (only most important eigenvectors)
	U = np.matmul(A, V)

	return U, m

################################################################################
# Calculate minimum number of eigenvectors needed to capture min_variance
#################################################################################
def reduce_number_of_eigenvectors(eigenvalues_training, min_variance):
	eigenvalues_training_len = len(eigenvalues_training)
	eigenvalues_training_sum = np.sum(eigenvalues_training)
	for k in range(eigenvalues_training_len):
		v = np.sum(eigenvalues_training[0:k]) / eigenvalues_training_sum
		if v >= min_variance:
			return k + 1 # Add one because k count starts at 0

################################################################################
# Extract and return features from each image in images (a.k.a. Projection)
#################################################################################
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
#################################################################################	
def normalize(data):
	# Column-wise subtract the mean and divide by the std deviation
	rows, columns = data.shape
	for r in range(rows):
		data[r] = (data[r] - (data[r]).mean(0)) / np.std(data[r])
		
	return data

def lda(img_training, classes_training, img_testing, num_of_people, num_of_faces_per_person, z):	
	# Step 1 - calculate separability between different classes (distance between means of different classes)
	# "Between-Class Matrix (Sb)
	# Step 2 - calculate distance between means and the samples of each class
	# "Within-Class Matrix (Sw)
	z = img_training.shape[0]
	Sb, Sw = lda_sb_and_sw(img_training, classes_training, num_of_people, num_of_faces_per_person, z)
	
	# Step 3 - Construct the lower dimensional space (Vk) that 
	# Maximizes Between-Class Matrix and minimizes Within-Class Matrix
	Swt = np.transpose(Sw)
	SwtSb = np.dot(Swt, Sb)
	eigenvalues, eigenvectors = np.linalg.eigh(SwtSb)
	# Decide how many eigenvectors are enough to represent variance in our training set - at least 95 % variance
	k = reduce_number_of_eigenvectors(eigenvalues, 0.95)
	# Dominant eigenvectors
	Vk = eigenvectors[:, 0 : k]
	Vk = np.transpose(Vk)
	
	# Step 4 - Project our original data into lower-dimensionalspace
	Vk_training = np.dot(Vk, img_training)
	Vk_testing = np.dot(Vk, img_testing)

	# Normalize data
	#Vk_training = normalize(Vk_training)	
	#Vk_testing = normalize(Vk_testing)
	
	return Vk_training, Vk_testing

def lda_sb_and_sw(img_training, classes_training, num_of_people, num_of_faces_per_person, z):
	# Compute mean of each class
	img_rows, img_columns = img_training.shape
	m = np.zeros((num_of_people, img_columns))
	classes_training = classes_training.astype(np.int)
	for r in range(img_rows):
		m[classes_training[r] - 1] = m[classes_training[r] - 1] + img_training[r]
	for r in range(num_of_people):
		m[r] = m[r] / num_of_faces_per_person
	
	# Compute total mean of all data
	m_total = (img_training.mean(0))[0]
	
	# Calculate separability between different classes (distance between means of different classes)
	# "Between-Class Matrix (Sb)
	Sb = np.zeros((z, z))
	for r in range(num_of_people):
		class_mean_minus_total = m[r] - m_total
		dot_product = np.dot(class_mean_minus_total, np.transpose(class_mean_minus_total))
		Sb = Sb + dot_product * float(5)
	
	# Calculate distance between means and the samples of each class
	# "Within-Class Matrix (Sw)	
	Sw = np.zeros((z, z))
	for r in range(num_of_people):
		mean = m[r]
		for img_r in range(img_rows): 
			img_minus_class_mean = img_training[img_r] - mean
			dot_product = np.dot(img_minus_class_mean, np.transpose(img_minus_class_mean))
			Sw = Sw + dot_product * float(5)
	return Sb, Sw
	
################################################################################
# Main knn function
#################################################################################	
def knn(W_training, classes_training, W_testing, classes_testing):
	print "Testing KNN"
	
	rows_training, columns_training = W_testing.shape # every row is an image we have to test
	rows_testing, columns_testing = W_testing.shape # every row is an image we have to test

	total_correct = 0.0
	total_incorrect = 0.0

	# For every reduced dimension image in W testing - check that it correctly classifies
	for r_testing in range(rows_testing):
		img_reduced = W_testing[r_testing]
		distances = np.zeros(rows_training)
		for r_training in range(rows_training):
			distances[r_training] = distance(img_reduced, W_training[r_training])
		# Check what this image should be classified as based on minimum distance
		classification = classes_training[np.argmin(distances)]
		# Check if the classification is correct
		if classification == classes_testing[r_testing]:
			total_correct = total_correct + 1.0
		else:
			total_incorrect = total_incorrect + 1.0
		
	print "\nTotal correct classifications: {}.\n".format(total_correct)
	print "\nTotal incorrect classifications: {}.\n".format(total_incorrect)
	accuracy = ((total_correct / (total_correct + total_incorrect)) * 100) # Accuracy
	print "\nAccuracy: {}%.\n".format(accuracy)
	return accuracy	

################################################################################
# Calculate distance between values of two lists of the same size 
#################################################################################
def distance(list1, list2):
	list_len = len(list1)
	residual_squared_sum = 0
	for i in range(list_len):
		residual_squared_sum = residual_squared_sum + math.pow(list1[i] - list2[i], 2)
	
	return math.sqrt(residual_squared_sum)
	
def main():

	# ----------------------------------------------
	# READ IN DATA
	# ----------------------------------------------	
	
	print "\nCreate Training and testing data..."	

	img_height = 112
	img_width = 92
	num_of_pixels = img_height * img_width
	num_of_people = 40
	num_of_faces_per_person = 10
	num_of_faces = num_of_people * num_of_faces_per_person
	images = np.zeros((num_of_faces, num_of_pixels)) # all images, each as row vector
	classes = np.zeros(num_of_faces) # all classes
	
	if (len(sys.argv) > 1):
		if (sys.argv[1] != "-resize"):
			print "\nAll of the images will be resized form (112x92) to (56x46)...\n"
	# Iterate through facial images and separate them into training and testing matrixes (80 : 20 ratio) with each row being a flat version of image
	counter = 0
	for i in range(1, num_of_people + 1, 1):
		for j in range(1, num_of_faces_per_person + 1, 1):
			# Read image
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm", 0) * (1.0 / 255.0)
			if (len(sys.argv) > 1):
				if (sys.argv[1] != "-resize"):
					img = cv2.resize(img, (56, 46)) 
			rows, columns = img.shape
			# 2D image matrix to 1D row vector
			images[counter] = (np.array(img)).flatten()
			classes[counter] = i
			counter = counter + 1


	# Convert images to type double
	images = images.astype(np.double)
	
	# ----------------------------------------------
	# BEGIN FIVE-FOLD CROSS VALIDATION PROCEDURE
	# ----------------------------------------------

	print "\nCreate Training and testing data groupd for 5-fold cross validation..."		
	
	# Shuffle images into 5 equal-size groups with equal number of images from each class
	img_group_0 = np.zeros((num_of_faces / 5, num_of_pixels))
	img_group_1 = np.zeros((num_of_faces / 5, num_of_pixels))
	img_group_2 = np.zeros((num_of_faces / 5, num_of_pixels))
	img_group_3 = np.zeros((num_of_faces / 5, num_of_pixels))
	img_group_4 = np.zeros((num_of_faces / 5, num_of_pixels))
	class_group_0 = np.zeros(num_of_faces / 5)
	class_group_1 = np.zeros(num_of_faces / 5)
	class_group_2 = np.zeros(num_of_faces / 5)
	class_group_3 = np.zeros(num_of_faces / 5)
	class_group_4 = np.zeros(num_of_faces / 5)
	counter = 0	
	for i in range(num_of_people):
		rand_indexes = random.sample(range(0, 10), 10)
		# group 0
		img_group_0[counter] = images[i*10 + rand_indexes[0]]
		img_group_0[counter + 1] = images[i*10 + rand_indexes[1]]
		class_group_0[counter] = i + 1
		class_group_0[counter + 1] = i + 1
		# group 1
		img_group_1[counter] = images[i*10 + rand_indexes[2]]
		img_group_1[counter + 1] = images[i*10 + rand_indexes[3]]
		class_group_1[counter] = i + 1
		class_group_1[counter + 1] = i + 1
		# group 2
		img_group_2[counter] = images[i*10 + rand_indexes[4]]
		img_group_2[counter + 1] = images[i*10 + rand_indexes[5]]
		class_group_2[counter] = i + 1
		class_group_2[counter + 1] = i + 1
		# group 3
		img_group_3[counter] = images[i*10 + rand_indexes[6]]
		img_group_3[counter + 1] = images[i*10 + rand_indexes[7]]
		class_group_3[counter] = i + 1
		class_group_3[counter + 1] = i + 1
		# group 4
		img_group_4[counter] = images[i*10 + rand_indexes[8]]
		img_group_4[counter + 1] = images[i*10 + rand_indexes[9]]
		class_group_4[counter] = i + 1
		class_group_4[counter + 1] = i + 1
		counter = counter + 2

	image_groups = [img_group_0, img_group_1, img_group_2, img_group_3, img_group_4]
	image_classes_groups = [class_group_0, class_group_1, class_group_2, class_group_3, class_group_4]

	print "\nRunning KNNs..."		
	
	# Use different unique combinations of the 5 groups to perform training and testing - average accuracy
	combinations = set(itertools.permutations([0, 1, 2, 3, 4]))
	average_accuracy_pca_knn = 0
	accuracies_pca_knn = np.zeros(120)
	average_accuracy_lda_knn = 0
	accuracies_lda_knn = np.zeros(120)
	average_accuracy_pca_lda_knn = 0
	accuracies_pca_lda_knn = np.zeros(120)
	counter = 0
	for combo in combinations:
		print "\n----------------------------------------------------------------------"
		print "Running combination {}. This is {} out of {}.".format(combo, counter + 1, 120)
		print "----------------------------------------------------------------------\n"
		# Training data
		img_training = np.concatenate((image_groups[combo[0]], image_groups[combo[1]], image_groups[combo[2]], image_groups[combo[3]]))
		# Convert image matrix to eigenfaces
		classes_training = np.concatenate((image_classes_groups[combo[0]], image_classes_groups[combo[1]], image_classes_groups[combo[2]], image_classes_groups[combo[3]]))
		# Testing data
		img_testing = np.array(image_groups[combo[4]], np.double)
		classes_testing = np.array(image_classes_groups[combo[4]])
		# Run PCA		
		print "\nPCA + KNN...\n"
		# Feature extraction using pca
		W_training, W_testing = pca(img_training, img_testing)
		# Face recognition using KNN
		accuracies_pca_knn[counter]  = knn(W_training, classes_training, W_testing, classes_testing)
		'''
		# Test to compare accuracy to sklearn algorithm
		knn = KNeighborsClassifier(n_neighbors = 1)
		knn.fit(W_training, classes_training)
		pred = knn.predict(W_testing)
		rows_testing, cols_testing = W_testing.shape
		total_correct = 0.0
		total_incorrect = 0.0	
		for r in range(rows_testing):
			# Check if the classification is correct
			if pred[r] == classes_testing[r]:
				total_correct = total_correct + 1.0
			else:
				total_incorrect = total_incorrect + 1.0
		accuracy = ((total_correct / (total_correct + total_incorrect)) * 100) # Accuracy
		print "\nAccuracy (sklearn): {}%.\n".format(accuracy)
		break
		'''
		
		# Run LDA
		#print "\nLDA + KNN...\n"
		z = img_training.shape[0]
		Vk_training, Vk_testing = lda(img_training, classes_training, img_testing, num_of_people, num_of_faces_per_person, z)
		
		
		# Run PCA + LDA
		#print "\nPCA + LDA + KNN...\n"
		accuracies_lda_knn[counter]  = knn(Vk_training, classes_training, Vk_testing, classes_testing)
		
		break
		
		counter = counter + 1
		
	#print "\nCombinations:\n"
	#print combinations 
	print "\nPCA + KNN accuracies:\n"
	print accuracies_pca_knn
	print "\nLDA + KNN accuracies:\n"
	print accuracies_lda_knn
	print "\nPCA + LDA + KNN accuracies:\n"
	print accuracies_pca_lda_knn
	average_accuracy_pca_knn = sum(accuracies_pca_knn) / 120
	print "\nAverage PCA + KNN accuracy: {}%.\n".format(average_accuracy_pca_knn)
	average_accuracy_lda_knn = sum(accuracies_lda_knn) / 120
	print "\nAverage LDA + KNN accuracy: {}%.\n".format(average_accuracy_lda_knn)
	average_accuracy_pca_lda_knn = sum(accuracies_pca_lda_knn) / 120
	print "\nAverage PCA + LDA + KNN accuracy: {}%.\n".format(average_accuracy_pca_lda_knn)
	
if __name__ == "__main__":
	main()