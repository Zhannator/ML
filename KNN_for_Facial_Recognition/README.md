KNN (k = 1) for face recognition

Overview:

Face recognition is implemented using k-nearest neighbors algorithm with pca, lda, and pca+lda preprocessing. Design is validated via five-fold cross validation and average accuracy is reported for each implementation. This code can also be run on images with reduced size.

Dataset:

The Database of Faces is used to perform all of the training and testing in this implementation. There are ten different images of each of 40 distinct subjects. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10). This database is made available by AT&T Laboratories Cambridge and can be downloaded here: http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html.

Prerequisites:

    python 2.7
    sys
	cv2
    numpy
    math
    random
    itertools

Running:

PCA + KNN: python knn_main.py -pca
PCA + KNN (resized images): python knn_main.py -pca -resize
LDA + KNN: python knn_main.py -lda
PCA + LDA + KNN: python knn_main.py -pca+lda

Function blocks:

--- main()

    Reads in images
    Breaks images into 5 image groups for five-fold cross validation
    Iterates through unique combinations of 5 image groups and calls (pca + knn), (lda + knn), or (pca + lda + knn) to train on and test on these unique combinations based on the input
    Reports average accuracies for (pca + knn), (lda + knn), or (pca + lda + knn) implementations

--- knn_1(W_training, classes_training, W_testing, classes_testing)

    Computes k-nearest neighbor (k = 1) classifications for testing data
    Returns accuracy

--- distance(list1, list2)

    Calculate distance between values of two lists of the same size
    Returns distance between two lists

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

Average PCA + KNN accuracy: 95.51 %

Average PCA + KNN accuracy (resized):  95.12 %

Average LDA + KNN accuracy: 97.5 %

Average PCA + LDA + KNN accuracy: 94.33 %

All PCA + KNN accuracy: [ 92.5   92.5   93.75  96.25  92.5   93.75  92.5   97.5   97.5   93.75
  93.75  93.75  97.5   96.25  96.25  96.25  97.5   96.25  93.75  92.5   92.5
  92.5   97.5   97.5   92.5   97.5   92.5   97.5   92.5   97.5   96.25
  97.5   97.5   97.5   93.75  97.5   93.75  97.5   97.5   93.75  96.25
  97.5   96.25  93.75  97.5   97.5   96.25  97.5   92.5   97.5   96.25
  97.5   97.5   92.5   93.75  97.5   96.25  92.5   97.5   92.5   96.25
  96.25  97.5   92.5   93.75  97.5   96.25  97.5   97.5   93.75  93.75
  96.25  96.25  96.25  97.5   97.5   97.5   93.75  97.5   93.75  97.5
  93.75  97.5   97.5   97.5   92.5   97.5   96.25  92.5   97.5   97.5   97.5
  97.5   93.75  93.75  92.5   96.25  93.75  96.25  97.5   92.5   97.5
  96.25  97.5   92.5   97.5   93.75  93.75  92.5   97.5   96.25  93.75
  92.5   96.25  93.75  92.5   97.5   97.5   97.5   92.5 ]

All PCA + KNN (resized) accuracy: [ 93.75  93.75  97.5   95.    93.75  96.25  93.75  95.    95.    97.5
  96.25  96.25  95.    95.    95.    95.    96.25  95.    96.25  93.75
  93.75  93.75  95.    95.    93.75  95.    93.75  95.    93.75  95.    95.
  95.    95.    96.25  96.25  95.    96.25  95.    95.    96.25  95.    95.
  95.    96.25  95.    95.    95.    95.    93.75  95.    95.    95.    96.25
  93.75  95.    95.    95.    93.75  95.    93.75  95.    95.    96.25
  93.75  96.25  95.    95.    95.    96.25  96.25  96.25  95.    95.    95.
  95.    95.    96.25  96.25  95.    96.25  95.    96.25  95.    95.    93.75
  93.75  95.    95.    93.75  95.    96.25  96.25  95.    96.25  96.25
  93.75  95.    97.5   95.    95.    93.75  95.    95.    95.    93.75  95.
  96.25  97.5   93.75  95.    95.    96.25  93.75  95.    97.5   93.75  95.
  95.    95.    93.75]

All PCA + LDA + KNN accuracy: [ 91.25  91.25  91.25  98.75  90.    91.25  91.25  92.5   93.75  92.5   92.5
  92.5   97.5   98.75  98.75  97.5   93.75  98.75  92.5   91.25  90.    90.
  98.75  97.5   91.25  96.25  91.25  93.75  90.    96.25  98.75  97.5
  93.75  93.75  92.5   97.5   92.5   98.75  93.75  92.5   97.5   96.25
  98.75  92.5   93.75  92.5   97.5   92.5   91.25  97.5   97.5   98.75
  93.75  90.    91.25  96.25  98.75  91.25  97.5   90.    98.75  97.5
  93.75  90.    92.5   92.5   98.75  93.75  93.75  91.25  92.5   97.5
  98.75  98.75  97.5   93.75  93.75  92.5   93.75  92.5   96.25  92.5   97.5
  98.75  93.75  91.25  96.25  98.75  90.    96.25  93.75  93.75  97.5   92.5
  92.5   91.25  98.75  91.25  98.75  95.    90.    93.75  98.75  96.25  90.
  93.75  92.5   91.25  91.25  95.    97.5   92.5   91.25  98.75  91.25
  91.25  92.5   96.25  97.5   90.  ]