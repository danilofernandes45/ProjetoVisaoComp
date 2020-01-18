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

descriptor_list = np.array( [[ 0 for i in range(128)]] )
num_descrip = []

for tag in ["Full", "Empty"]:
    for i in range(1, 41):
        image = cv2.imread("Dataset/"+tag+"/img"+str(i)+".jpg", 0)
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
    for i in range(41, 51):
        image = cv2.imread("Dataset/"+tag+"/img"+str(i)+".jpg", 0)
        descriptors = features(image, extractor)[1]
        hist = np.zeros( n_clusters )
        for k in range( len( descriptors ) ):
            hist[ kmeans.predict( [ descriptors[k] ] ) ] += 1

        norm_hist = hist / len(descriptors)
        prediction = classifier.predict([norm_hist])[0]

        if( tag == "Full" and prediction == 0 or tag == "Empty" and prediction == 1 ):
             hit += 1

print(hit)
