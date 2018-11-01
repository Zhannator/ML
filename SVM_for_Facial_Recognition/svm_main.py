# Public libraries
import cv2
import numpy as np
from cvxopt import matrix, solvers
import math
import random
import itertools

def training(img_training, classes_training, num_of_people, polynomial_flag = False, polynomial_degree = 2.0):
	# Row image -> column images
	img_training_t = np.transpose(img_training)	
	rows, columns = img_training_t.shape

	# Constants
	c = 2.0

	# These variables will collect w's and b's for each class
	svm_w = np.zeros((num_of_people, rows))
	svm_b = np.zeros(num_of_people)	
	
	# Column to row images
	img_training_t = np.array(img_training_t, np.double)
	
	# Iterate through 40 classes
	for class_i in range(1, num_of_people + 1, 1):
		# Create label list where images in this class will be labeled as 1 and images not in this class will be labeled as -1
		labels = np.zeros(columns)
		labels_index = 0
		for my_c in classes_training:
			if my_c == class_i:
				labels[labels_index] = 1
			else:
				labels[labels_index] = -1
			labels_index = labels_index + 1

		# Labels has to be of type double for quadratic programming solver
		labels = np.array(labels, np.double)		

		# Calculate K
		K = matrix(np.zeros((columns, columns)))
		for i in range(columns):
			for j in range(columns):
				# Choose between linear and polynomial svm
				if polynomial_flag == True:
					K[i, j] = math.pow((np.dot(img_training_t[:, i], img_training_t[:, j]) + 1.0), polynomial_degree)
				else: # Linear and aslo default
					# For linear SVM, single K value is calculated using dot product of two column vectors from training data			
					K[i, j] = np.dot(img_training_t[:, i], img_training_t[:, j])
				
				

		# Calculate variables needed for quadratic programming solver
	        P = matrix(np.outer(labels, labels) * K)
		q = matrix(- np.ones(columns))
		G = matrix(np.vstack((np.diag(-np.ones(columns)), np.identity(columns))))
		h = matrix(np.hstack((np.zeros(columns), np.ones(columns) * c)))
		A = matrix(labels, (1, columns))
		b = matrix(np.zeros(1))		
		
		print "Running quadratic programming solver for class s{}...".format(class_i)
	
		# Calculate alfas using quadratic programming solver
		solvers.options['show_progress'] = False        	
		solution = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(solution['x'], np.double)		
		alpha_min = min(alphas)

		# Calculate w		
		w = np.sum(alphas * labels[:, None] * img_training, axis = 0)
		svm_w[class_i - 1] = w	
		# Calculate b
		non_zero_alpha_indexes = (np.where(alphas > alpha_min))[0]
		b = np.zeros(len(non_zero_alpha_indexes))
		b_index = 0
	    	for index in non_zero_alpha_indexes: # iterate through list of indexes where alpha > const
			b_dot_product = np.dot(img_training[index], np.transpose(w))
			b_temp = labels[index] - np.dot(img_training[index], w)
			b[b_index] = b_temp
			b_index = b_index + 1
		bias = b[0]
		svm_b[class_i - 1] = bias
	
	return svm_w, svm_b

def testing(img_testing, classes_testing, num_of_people, w, b):
	rows, columns = img_testing.shape # every row is an image we have to test

	total_correct = 0.0
	total_incorrect = 0.0

	# For every image in testing data - check that it correctly classifies
	for r in range(rows):
		img = img_testing[r]
		for i in range(num_of_people):
			test = np.dot(w[i], img) + b[i]
			current_class = classes_testing[r]
			if ((test > 0) & (i + 1 == current_class)): # True Positive - Classified Correctly
				total_correct = total_correct + 1.0
				#print "\nImage {} classified true positive for class {}.\n".format(r + 1, current_class)
			elif ((test <= 0) & (i + 1 != current_class)): # True Negative - Classified Correctly
				total_correct = total_correct + 1.0
				#print "\nImage {} classified true negative for class {}.\n".format(r + 1, current_class)
			elif ((test > 0) & (i + 1 != current_class)): # False Positive - Classified Incorrectly
				total_incorrect = total_incorrect + 1.0
				#print "\n--Image {} classified false positive for class {}.--\n".format(r + 1, current_class)		
			else: # False Negative - Classified Incorrectly
				total_incorrect = total_incorrect + 1.0
				#print "\n--Image {} classified false negative as class {}.--\n".format(r + 1, current_class)		
		
	print "\nTotal correct classifications: {}.\n".format(total_correct)
	print "\nTotal incorrect classifications: {}.\n".format(total_incorrect)
	accuracy = ((total_correct / (total_correct + total_incorrect)) * 100) # Accuracy
	print "\nAccuracy: {}%.\n".format(accuracy)
	return accuracy

def svm(num_of_people, img_training, classes_training, img_testing, classes_testing, polynomial_flag = False, polynomial_degree = 2.0):
	# ----------------------------------------------
	# TRAINING
	# ----------------------------------------------
	
	print "\nTraining..."

	w, b = training(img_training, classes_training, num_of_people, polynomial_flag, polynomial_degree) # w and b are in order of class (1 through 40)

	# ----------------------------------------------
	# TESTING
	# ----------------------------------------------
		
	print "\nTesting..."

	accuracy = testing(img_testing, classes_testing, num_of_people, w, b)
	
	return accuracy

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

	# Iterate through facial images and separate them into training and testing matrixes (80 : 20 ratio) with each row being a flat version of image
	counter = 0
	for i in range(1, num_of_people + 1, 1):
		for j in range(1, num_of_faces_per_person + 1, 1):
			# Read image
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm", 0)
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

	print "\nRunning SVMs..."		
	
	# Use different unique combinations of the 5 groups to perform training and testing - average accuracy
	combinations = set(itertools.permutations([0, 1, 2, 3, 4]))
	average_accuracy_linear = 0
	accuracies_linear = np.zeros(120)
	average_accuracy_quadratic = 0
	accuracies_quadratic = np.zeros(120)
	counter = 0
	for combo in combinations:
		print "\n----------------------------------------------------------------------"
		print "Running combination {}. This is {} out of {}.".format(combo, counter + 1, 120)
		print "----------------------------------------------------------------------\n"
		# Training data
		img_training = np.concatenate((image_groups[combo[0]], image_groups[combo[1]], image_groups[combo[2]], image_groups[combo[3]]))
		classes_training = np.concatenate((image_classes_groups[combo[0]], image_classes_groups[combo[1]], image_classes_groups[combo[2]], image_classes_groups[combo[3]]))
		# Testing data
		img_testing = np.array(image_groups[combo[4]], np.double)
		classes_testing = np.array(image_classes_groups[combo[4]])
		# Run linear SVM		
		print "\nLinear SVM...\n"		
		accuracies_linear[counter] = svm(num_of_people, img_training, classes_training, img_testing, classes_testing)
		# Run quadratic (degree = 2) SVM
		print "\nQuadratic SVM...\n"
		accuracies_quadratic[counter] = svm(num_of_people, img_training, classes_training, img_testing, classes_testing, True, 2.0)
		counter = counter + 1	
	
	print "\nCombinations:\n"
	print combinations 
	print "\nLinear accuracies:\n"
	print accuracies_linear
	print "\nQuadratic accuracies:\n"
	print accuracies_quadratic
	average_accuracy_linear = sum(accuracies_linear) / 120
	print "\nAverage linear accuracy: {}%.\n".format(average_accuracy_linear)
	average_accuracy_quadratic = sum(accuracies_quadratic) / 120
	print "\nAverage quadratic accuracy: {}%.\n".format(average_accuracy_quadratic)

if __name__ == "__main__":
	main()