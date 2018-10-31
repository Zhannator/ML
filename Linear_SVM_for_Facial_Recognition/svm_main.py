# Public libraries
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import math

def training(img_training, num_of_people, polynomial_flag = False, polynomial_degree = 2.0):
	# Row image -> column images
	img_training_t = np.transpose(img_training)	
	rows, columns = img_training_t.shape

	# Constants
	c = 2.0

	# These variables will collect w's and b's for each class
	svm_w = []
	svm_b = []	
	
	# Column to row images
	img_training_t = np.array(img_training_t, np.double)
	
	# Iterate through 40 classes
	for class_i in range(1, num_of_people + 1, 1):
		# Create label list where images in this class will be labeled as 1 and images not in this class will be labeled as -1
		labels = []
		b = 8 # b depends on ratio of training to testing data, in this case we are doing 80 : 20, so 8 out of 10 pictures are used for training and will be labeled as 1
		# Insert non-class labels before class labels
		labels.extend([-1] * ((class_i - 1) * 8))
		# Insert class labels
		labels.extend([1] * 8) # depends on ratio of training to testing data, in this case we are doing 80 : 20, so 8 out of 10 pictures are used for training and will be labeled as 1
		# Insert non-class labels after class labels
		labels.extend([-1] * ((num_of_people - class_i) * 8))
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
		#print "\nP: \n"
		#print P
		q = matrix(- np.ones(columns))
		#print "\nq: \n"
		#print q

		G = matrix(np.vstack((np.diag(-np.ones(columns)), np.identity(columns))))
		#print "\nG: \n"
		#print G
		h = matrix(np.hstack((np.zeros(columns), np.ones(columns) * c)))
		#print "\nh: \n"
		#print h

		A = matrix(labels, (1, columns))
		#print "\nA: \n"
		#print A

		b = matrix(np.zeros(1))
		#print "\nb: \n"
		#print b		

		
		print "Running quadratic programming solver for class s{}...".format(class_i)
	

		# Calculate alfas using quadratic programming solver

        	solution = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(solution['x'], np.double)		
		alpha_min = min(alphas)
		#print alphas

		# Calculate w		
		w = np.sum(alphas * labels[:, None] * img_training, axis = 0)
		#print "\nTHIS IS W:\n"	
		#print w
		#print len(w)
		svm_w.append(w)	
		# Calculate b
		non_zero_alpha_indexes = (np.where(alphas > alpha_min))[0]
		#print non_zero_alphas	
		b = []
		#print labels
		#print img_training
		#print img_training[0]
		#print non_zero_alpha_indexes
	    	for index in non_zero_alpha_indexes: # iterate through list of indexes where alpha > const
			b_dot_product = np.dot(img_training[index], np.transpose(w))
			#print img_training[index]
			#print w
			#exit(0)
			b_temp = labels[index] - np.dot(img_training[index], w)
			b.append(b_temp)
		#print "\nTHIS IS B:\n"		
		#print b
		bias = b[0]
		#print bias
		svm_b.append(bias)
		#exit(0)
	
	return svm_w, svm_b

def testing(img_testing, num_of_people, w, b):
	rows, columns = img_testing.shape # every row is an image we have to test
	#print rows
	#print columns
	#print b

	total_correct = 0.0
	total_incorrect = 0.0	

	# For every image in testing data - check that it correctly classifies
	for r in range(rows):
		img = img_testing[r]
		for i in range(num_of_people):
			#print w[i]
			test = np.dot(w[i], img) + b[i]
			#print test
			current_class = int(r / 2) + 1
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
	print "\nOverall accuracy is {}%.\n".format(accuracy)
	return accuracy

def svm(num_of_people, img_training, img_testing):
	# ----------------------------------------------
	# TRAINING
	# ----------------------------------------------
	
	print "\nTraining..."

	w, b = training(img_training, num_of_people) # w and b are in order of class (1 through 40)
	
	#print w
	#print w.shape
	#print b

	# ----------------------------------------------
	# TESTING
	# ----------------------------------------------
		
	print "\nTesting..."

	accuracy = testing(img_testing, num_of_people, w, b)

def main():

	# ----------------------------------------------
	# CREATE TRAINING AND TESTING DATA
	# ----------------------------------------------	
	
	print "\nCreate Training and testing data..."	

	img_height = 112
	img_width = 92
	num_of_pixels = img_height * img_width
	num_of_people = 40
	num_of_faces_per_person = 10
	num_of_faces = num_of_people * num_of_faces_per_person
	img_training = [] # data matrix for training (80%) classes = [1, ..., 1, 2, ..., 40, ..., 40]
	img_testing = [] # data matrix for testing (20%) classes = [1 , 1, 2, ..., 39, 39, 40, 40]
	
	# Iterate through facial images and separate them into training and testing matrixes (80 : 20 ratio) with each row being a flat version of image
	for i in range(1, num_of_people + 1, 1):
		for j in range(1, num_of_faces_per_person + 1, 1):
			# Read image
			img = cv2.imread("data/s" + str(i) + "/" + str(j) + ".pgm", 0)
			rows, columns = img.shape
			# 2D image matrix to 1D row vector
			img = (np.array(img)).flatten()
			if j < 9:
				img_training.append(img)
			else:
				img_testing.append(img)

	# Convert images to type double
	img_training = (np.array(img_training)).astype(np.double)
	img_testing = (np.array(img_testing)).astype(np.double)
	
	# ----------------------------------------------
	# BEGIN FIVE-FOLD CROSS VALIDATION PROCEDURE
	# ----------------------------------------------
	svm(num_of_people, img_training, img_testing)

if __name__ == "__main__":
	main()