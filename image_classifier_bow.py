import cv2

# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 800)
kmeans.fit(descriptor_list)

preprocessed_image = []
for image in images:
    image = gray(image)
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
        histogram = build_histogram(descriptor, kmeans)
        preprocessed_image.append(histogram)

from sklearn.neighbors import NearestNeighbors

data = cv2.imread(image_path)
data = gray(data)
keypoint, descriptor = features(data, extractor)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors = 20)
neighbor.fit(preprocess_image)
dist, result = neighbor.kneighbors([histogram])
