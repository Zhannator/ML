Support vector machines for face recognition

Overview:

Face recognition is implemented using linear and polynomial support vector machines (SVM) algorithms with one vs rest methodology. Design is validated via five-fold cross validation and average accuracy is reported for each implementation, linear and polynomial (degree = 2). 

Dataset:

The Database of Faces is used to perform all of the training and testing in this implementation. There are ten different images of each of 40 distinct subjects. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10). This database is made available by AT&T Laboratories Cambridge and can be downloaded here: http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html.

Prerequisites:

-	python 2.7
-	cv2
-	numpy
-	cvxopt
-	math
-	random
-	itertools

Running:

python svm_main.py

Function blocks:

---	main()
-	Reads in images
-	Breaks images into 5 image groups for five-fold cross validation
-	Iterates through unique combinations of 5 image groups and calls linear and polynomial svm() to train on and test on these unique combinations
-	Reports average accuracies for linear and polynomial implementations

---	svm(num_of_people, img_training, classes_training, img_testing, classes_testing, polynomial_flag = False, polynomial_degree = 2.0)
-	Calls training() and testing() function blocks
-	Returns accuracy

---	training(img_training, classes_training, num_of_people, polynomial_flag = False, polynomial_degree = 2.0)
-	Calculates a model (w and b) for every class using quadratic programming solver
-	Returns a list of w’s and b’s each corresponding to one of the classes (40 people)

---	testing(img_testing, classes_testing, num_of_people, w, b)
-	Tests each test images against every class (40 people) and reports on total of correctly and incorrectly classified images
-	Returns accuracy

Results:

Average linear accuracy: 84.6515625%

Average quadratic (degree = 2) accuracy: 95.125%

Group combinations:
[(0, 3, 2, 4, 1), (2, 0, 3, 4, 1), (1, 3, 2, 4, 0), (1, 2, 0, 3, 4), (0, 3, 4, 2, 1), (2, 4, 1, 3, 0), (3, 4, 0, 2, 1), (0, 4, 3, 1, 2), (0, 1, 4, 3, 2), (1, 3, 4, 2, 0), (4, 2, 1, 3, 0), (2, 1, 3, 4, 0), (1, 4, 0, 2, 3), (0, 3, 2, 1, 4), (2, 0, 1, 3, 4), (0, 2, 3, 1, 4), (4, 0, 1, 3, 2), (2, 1, 0, 3, 4), (3, 1, 2, 4, 0), (3, 2, 0, 4, 1), (2, 3, 4, 0, 1), (4, 2, 0, 3, 1), (2, 4, 1, 0, 3), (2, 1, 4, 0, 3), (3, 0, 2, 4, 1), (2, 1, 0, 4, 3), (0, 2, 3, 4, 1), (4, 3, 1, 0, 2), (0, 2, 4, 3, 1), (0, 4, 2, 1, 3), (1, 0, 3, 2, 4), (1, 2, 4, 0, 3), (1, 3, 4, 0, 2), (3, 0, 1, 4, 2), (4, 1, 2, 3, 0), (4, 1, 2, 0, 3), (2, 4, 3, 1, 0), (2, 0, 4, 1, 3), (1, 4, 3, 0, 2), (2, 3, 1, 4, 0), (0, 1, 2, 3, 4), (0, 2, 1, 4, 3), (2, 3, 0, 1, 4), (4, 3, 1, 2, 0), (4, 1, 3, 0, 2), (0, 4, 1, 3, 2), (0, 2, 1, 3, 4), (0, 3, 1, 4, 2), (3, 0, 4, 2, 1), (1, 4, 2, 0, 3), (3, 1, 0, 2, 4), (1, 0, 2, 4, 3), (3, 4, 0, 1, 2), (4, 0, 3, 2, 1), (1, 4, 3, 2, 0), (0, 1, 4, 2, 3), (3, 0, 2, 1, 4), (0, 4, 2, 3, 1), (0, 2, 4, 1, 3), (2, 4, 3, 0, 1), (1, 3, 2, 0, 4), (3, 2, 0, 1, 4), (4, 1, 0, 3, 2), (2, 0, 4, 3, 1), (3, 1, 4, 2, 0), (1, 0, 3, 4, 2), (2, 0, 3, 1, 4), (0, 1, 3, 4, 2), (3, 1, 0, 4, 2), (2, 1, 4, 3, 0), (3, 2, 1, 4, 0), (0, 3, 1, 2, 4), (2, 1, 3, 0, 4), (1, 2, 3, 0, 4), (0, 1, 2, 4, 3), (1, 0, 4, 3, 2), (4, 0, 3, 1, 2), (3, 4, 2, 1, 0), (4, 3, 0, 1, 2), (4, 2, 3, 1, 0), (2, 4, 0, 1, 3), (4, 3, 2, 1, 0), (1, 2, 0, 4, 3), (1, 0, 4, 2, 3), (1, 4, 0, 3, 2), (4, 2, 3, 0, 1), (0, 4, 1, 2, 3), (1, 0, 2, 3, 4), (4, 3, 2, 0, 1), (4, 2, 1, 0, 3), (3, 0, 4, 1, 2), (3, 1, 4, 0, 2), (2, 0, 1, 4, 3), (3, 2, 4, 1, 0), (2, 3, 4, 1, 0), (4, 3, 0, 2, 1), (0, 1, 3, 2, 4), (1, 2, 4, 3, 0), (2, 3, 1, 0, 4), (4, 2, 0, 1, 3), (4, 0, 2, 3, 1), (0, 3, 4, 1, 2), (1, 3, 0, 2, 4), (4, 1, 0, 2, 3), (2, 4, 0, 3, 1), (3, 4, 1, 0, 2), (3, 4, 1, 2, 0), (1, 4, 2, 3, 0), (3, 4, 2, 0, 1), (4, 0, 1, 2, 3), (3, 0, 1, 2, 4), (4, 1, 3, 2, 0), (0, 4, 3, 2, 1), (3, 2, 1, 0, 4), (1, 2, 3, 4, 0), (3, 2, 4, 0, 1), (1, 3, 0, 4, 2), (4, 0, 2, 1, 3), (3, 1, 2, 0, 4), (2, 3, 0, 4, 1)]

All linear accuracies (1 per group combination):
[85.      83.53125 82.875   81.78125 85.      84.375   88.1875  85.21875
 85.21875 82.875   85.9375  84.375   80.875   84.8125  83.78125 84.8125
 85.21875 83.78125 86.375   88.1875  83.53125 85.78125 84.6875  84.6875
 88.1875  84.6875  85.      85.21875 85.      85.09375 81.78125 80.875
 82.      85.90625 85.9375  85.71875 84.375   84.6875  82.      84.375
 84.8125  85.09375 83.78125 85.9375  85.21875 85.21875 84.8125  85.21875
 88.1875  80.875   85.875   80.875   85.90625 85.78125 82.875   85.09375
 85.875   85.      85.09375 83.53125 81.78125 85.875   85.21875 83.53125
 86.375   82.      83.78125 85.21875 85.90625 84.375   86.375   84.8125
 83.78125 81.78125 85.09375 82.      85.21875 86.375   85.21875 85.9375
 84.6875  85.9375  80.875   80.875   82.      85.78125 85.09375 81.78125
 85.78125 85.71875 85.90625 85.90625 84.6875  86.375   84.375   85.78125
 84.8125  82.875   83.78125 85.71875 85.78125 85.21875 81.78125 85.71875
 83.53125 85.90625 86.375   82.875   88.1875  85.71875 85.875   85.9375
 85.      85.875   82.875   88.1875  82.      85.71875 85.875   83.53125]

All quadratic accuracies (1 per group combination):
[95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125
 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125 95.125]