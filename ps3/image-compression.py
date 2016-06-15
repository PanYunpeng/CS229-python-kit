import random

from skimage import io
import numpy as np

A = io.imread('data/mandrill-large.tiff')
io.imshow(A)

small = io.imread('data/mandrill-small.tiff')
x, y, _ = small.shape
pixels = small.astype(np.float).reshape((x * y, 3))

# Run K-mean clustering with k= 16
k = 16
def euclidean_distance(u, v):
    raise NotImplementedError()

def kmeans(data, k, dist=euclidean_distance, max_iter=100):
    ''' Run K-means clustering on the data set

    Returns: K _collections_ of data points
    '''
    raise NotImplementedError()

centroids, labels = kmeans(pixels, k)

# Compress the large mandrill image
X, Y, _ = A.shape
M = X * Y
A_pixels = A.reshape((M, 3))

# turns out when doing the distance calculation we need to do so in signed
# integer space, otherwise there will be underflow
centroids_i = centroids.astype('int')

A_pixel_labels = [ np.argmin(euclidean_distance(centroids_i, A_pixels[x]))
                   for x in range(M) ]
new_A_pixels = np.array([centroids[label] for label in A_pixel_labels])
new_A = new_A_pixels.reshape((X, Y, 3))

io.imshow(new_A)
io.imsave('mandrill-large-compressed.png', new_A)
