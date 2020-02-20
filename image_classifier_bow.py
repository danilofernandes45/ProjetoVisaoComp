import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm

n_clusters = 100

# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def handOut():
    descriptor_list = np.array( [[ 0 for i in range(128)]] )
    num_descrip = []

    for tag in ["Full", "Empty"]:
        for i in range(1, 41):
            image = cv2.imread("Dataset/"+tag+"/img"+str(i)+".png", 0)
            descriptors = features(image, extractor)[1]
            descriptor_list = np.append(descriptor_list, descriptors, axis = 0)
            num_descrip.append( descriptors.shape[0] )

    descriptor_list = np.delete(descriptor_list, 0, axis = 0)

    # image = cv2.imread("Dataset/Full/img2.jpg", 0)
    # print(features(image, extractor)[1][0])
    # print(descriptor_list[8343])

    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit( descriptor_list )

    print(kmeans.labels_[0:10])

    hist_images = []
    idx = 0
    for num in num_descrip:
        hist = np.zeros( n_clusters )
        for k in range(num):
            hist[kmeans.labels_[idx]] += 1
            idx += 1
        norm_hist = hist / num
        hist_images.append( norm_hist )

    hist_images = np.array(hist_images)
    classes = np.zeros(80)
    classes[40:81] = 1

    classifier = svm.SVC()
    classifier.fit(hist_images, classes)

    #Test phase
    hit = 0
    for tag in ["Full", "Empty"]:
        for i in range(41, 53):
            image = cv2.imread("Dataset/"+tag+"/img"+str(i)+".png", 0)
            descriptors = features(image, extractor)[1]
            hist = np.zeros( n_clusters )
            for k in range( len( descriptors ) ):
                hist[ kmeans.predict( [ descriptors[k] ] ) ] += 1

            norm_hist = hist / len(descriptors)
            prediction = classifier.predict([norm_hist])[0]

            if( tag == "Full" and prediction == 0 or tag == "Empty" and prediction == 1 ):
                 hit += 1

    print(hit)

def leaveOneOut():

    SIZE = 52

    total_descriptor_list = np.array( [[ 0 for i in range(128)]] )
    total_start_descrip = [0]
    total_num_descrip = []

    for tag in ["Full", "Empty"]:
        for i in range(1, SIZE+1):
            image = cv2.imread("Dataset/"+tag+"/img"+str(i)+".png", 0)
            descriptors = features(image, extractor)[1]
            total_descriptor_list = np.append(total_descriptor_list, descriptors, axis = 0)
            total_num_descrip.append( descriptors.shape[0] )
            total_start_descrip.append( total_start_descrip[-1] + descriptors.shape[0] )

    total_descriptor_list = np.delete(total_descriptor_list, 0, axis = 0)
    total_num_descrip = np.array(total_num_descrip)
    total_start_descrip = np.array(total_start_descrip)

    total_classes = np.zeros(2*SIZE)
    total_classes[SIZE:2*SIZE] = 1

    hit = 0

    for i in range( SIZE ):

        emp_begin = total_start_descrip[i]
        emp_end = emp_begin + total_num_descrip[i]

        full_begin = total_start_descrip[SIZE+i]
        full_end = full_begin + total_num_descrip[SIZE+i]

        descriptor_list = np.delete(total_descriptor_list, np.append(range(emp_begin, emp_end), range(full_begin, full_end)), axis = 0)
        num_descrip = np.delete(total_num_descrip, [i, SIZE+i], axis = 0)

        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit( descriptor_list )

        hist_images = []
        idx = 0
        for num in num_descrip:
            hist = np.zeros( n_clusters )
            for k in range(num):
                hist[kmeans.labels_[idx]] += 1
                idx += 1
            norm_hist = hist / num
            hist_images.append( norm_hist )

        hist_images = np.array(hist_images)

        classes = np.delete(total_classes, [i, SIZE+i], axis = 0)

        classifier = svm.SVC()
        classifier.fit(hist_images, classes)

        for n in range(102):
            print(classifier.predict([hist_images[n]]))

        hist = np.zeros( n_clusters )
        for k in range( total_num_descrip[i] ):
            hist[ kmeans.predict( [ total_descriptor_list[emp_begin + k] ] ) ] += 1

        norm_hist = hist / total_num_descrip[i]
        prediction = classifier.predict([norm_hist])[0]

        if( prediction == total_classes[i] ):
             hit += 1

        for k in range( total_num_descrip[i+SIZE] ):
            hist[ kmeans.predict( [ total_descriptor_list[full_begin + k] ] ) ] += 1

        norm_hist = hist / total_num_descrip[i+SIZE]
        prediction = classifier.predict([norm_hist])[0]

        if( prediction == total_classes[i+SIZE] ):
             hit += 1
        print(i)
        print(hit)

leaveOneOut()
