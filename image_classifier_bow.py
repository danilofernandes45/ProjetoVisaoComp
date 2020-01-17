import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

descriptor_list = []
for i in range(1, 41):
    image = cv2.imread("Dataset/Full/img"+str(i)+".jpg", 0)
    descriptor_list.append( features(image, extractor)[1] )
for i in range(1, 41):
    image = cv2.imread("Dataset/Empty/img"+str(i)+".jpg", 0)
    descriptor_list.append( features(image, extractor)[1] )

kmeans = KMeans(n_clusters = 100)
kmeans.fit(descriptor_list)

print(kmeans[1])

preprocessed_image = []
for descriptor in descriptor_list:
    if (descriptor is not None):
        histogram = build_histogram(descriptor, kmeans)
        preprocessed_image.append(histogram)



# data = cv2.imread(image_path)
# data = gray(data)
# keypoint, descriptor = features(data, extractor)
# histogram = build_histogram(descriptor, kmeans)
# neighbor = NearestNeighbors(n_neighbors = 20)
# neighbor.fit(preprocess_image)
# dist, result = neighbor.kneighbors([histogram])
