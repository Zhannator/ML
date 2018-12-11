Neural Network for Handwritten Character Recognition

Overview:

Handwritten character recognition is implemented using artificial neural network with pca + lda preprocessing. The neural network implemented in this project is composed of 3 layers, one input, one hidden and one output. This code can also be run on images with reduced size.

Dataset:

The Database of English Handwritten Characters is used to perform all of the training and testing in this implementation. There are 55 different images of each of 62 distinct characters. The size of each image is 900x1200 pixels. The images are organised in 62 directories (one for each character), which have names of the form SampleX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form SampleX-Y.pgm, where Y is the image number for that character (between 1 and 55). This database is made available by University of Surrey and can be downloaded here: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz.

Prerequisites:

python 2.7
sys
cv2
numpy
math
random
itertools

Running:
Run using original images without PCA or LDA: python ann_main.py
Run using reduced size images without PCA or LDA: python ann_main.py -resize
Run using reduced size images with PCA: python ann_main.py -resize -pca
Run using reduced size images with LDA: python ann_main.py -resize -lda
Run using reduced size images with PCA and LDA: python ann_main.py -resize -pca -lda

Function blocks:

--- main()

    Calls read_data to read in images
    Calls PCA, LDA, and ANN functions
    Reports overall accuracy and saves ann output to output.txt 

--- read_data()

    Reads in images from data folder
    Splits images into training and testing images and assigns their respective classes
    Returns images_training, classes_training, images_testing, classes_testing

--- sigmoid(data, derivative = False)

    Activation function for ANN
    Can be used to calculate sigmoid (derivative = False) or its derivative (derivative = True)
 
--- ann_f(imgages, weights_0_1, weights_1_2)

    Forward part of ANN that can be used to compute 3 layers (input, hidden, output)
    Returns 3 layers

--- ann_b(classes, layer_0, layer_1, layer_2, weights_0_1, weights_1_2)

    Backpropagation part of ANN
    Computes error and adjusts weights
    Returns adjusted weights

--- output_error(expected_classes, actual_classes)

    Calculates difference between expected and actual classes
    Returns difference

-- ann_test(expected_classes, actual_classes)

    Uses expected and actual output to calculate ANN confidences and overall accuracy

--- pca(img_training, img_testing)

    Main pca function
	Calls pca_analysis, pca_extract_features, and pca_normalize
    Returns training and testing dataset in reduced dimension

--- pca_analysis(T)

    Calculate U (most important eigenvectors from AAt) by multiplying A and V (only most important eigenvectors)
    Returns U and average image from training dataset

--- pca_extract_features(U, images, m)

    Extracts and return features from each image in images (a.k.a. Projection)

--- pca_normalize(data)

    Normalizes data using mean and standard deviation

--- lda(img_training, classes_training, img_testing, num_of_people, num_of_faces_per_person, z)

    Main lda function
	Calls lda_sb_and_sw to compute Between-Class Matrix (Sb) and Within-Class Matrix (Sw)
	Constructs the lower dimensional space (Vk) that maximizes Between-Class Matrix and minimizes Within-Class Matrix
	Projects original data into lower-dimensional space
    Returns training and testing dataset in reduced dimension

--- lda_sb_and_sw(img_training, classes_training, num_of_people, num_of_faces_per_person, z)

    Calculates separability between different classes (distance between means of different classes) - Between-Class Matrix (Sb)
	Calculates distance between means and the samples of each class - Within-Class Matrix (Sw)
    Returns Sb and Sw

--- reduce_number_of_eigenvectors(eigenvalues_training, min_variance)

    Calculates and returns minimum number of eigenvectors needed to capture min_variance (k)

Results:
	Overall Accuracy = 