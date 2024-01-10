import numpy as np
import random

def euclidean(point, data):
  return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
  def __init__(self, k=2, max_iterations=500, tol=0.5):
    # number of clusters
    self.k = k
    # maximum number of iterations to perform
    # for updating the centroids
    self.max_iterations = max_iterations
    # tolerance level for centroid change after each iteration
    self.tol = tol
    # we will store the computed centroids 
    self.centroids = None

  def init_centroids(self, X):
    # this function initializes the centroids
    # by choosing self.k points from the dataset
    # Hint: you may want to use the np.random.choice function
    min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
    centroids = [random.uniform(min_, max_) for _ in range(self.k)]
    #centroids = np.random.choice(X, size=self.k, replace=False)
    return centroids

  def closest_centroid(self, X):
    # this function computes the distance (euclidean) between 
    # each point in the dataset from the centroids filling the values
    # in a distance matrix (dist_matrix) of size n x k
    # Hint: you may want to remember how we solved the warm-up exercise
    # in Programming module (Python_Numpy2 file)
    dist_matrix = np.zeros((X.shape[0], self.k))
    for i in range(self.k):
      dist_matrix[:, i] = euclidean(X, self.centroids[i])
    # after constructing the distance matrix, you should return
    # the index of minimal value per row
    # Hint: you may want to use np.argmin function
    return np.argmin(dist_matrix, axis=1)

  def update_centroids(self, X, label_ids):
    # this function updates the centroids (there are k centroids)
    # by taking the average over the values of X for each label (cluster)
    # here label_ids are the indices returned by closest_centroid function
    
    new_centroids = np.zeros((self.k, X.shape[1]))
    for i in range(self.k):
      cluster_points = X[label_ids == i]
      if len(cluster_points) > 0:
        new_centroids[i] = np.mean(cluster_points, axis=0)
      else:
        # If a cluster has no points assigned, keep the previous centroid.
        new_centroids[i] = self.centroids[i]
    return new_centroids

  def fit(self, X):
    # this is the main method of this class
    X = np.array(X)
    # we start by random centroids from our data
    self.centroids = self.init_centroids(X)

    not_converged = True
    i = 1 # keeping track of the iterations
    while not_converged and (i < self.max_iterations):
      current_labels = self.closest_centroid(X)
      new_centroids = self.update_centroids(X, current_labels)

      # count the norm between new_centroids and self.centroids
      # to measure the amount of change between 
      # old cetroids and updated centroids
      norm = np.linalg.norm(new_centroids - self.centroids)
      not_converged = norm > self.tol
      self.centroids = new_centroids
      i += 1
    self.labels = current_labels
    print(f'Converged in {i} steps')


class HierarchicalClustering:
  def __init__(self, nr_clusters, diss_func, linkage='single', distance_threshold=None):
    # nr_clusters is the number of clusters to find from the data
    # if distance_threshold is None, nr_clusters should be provided
    # and if distance_threshold is provided, then we stop 
    # forming clusters when we reach the specified threshold 
    # diss_func is the dissimilarity measure to compute the 
    # dissimilarity/distance between two data points
    self.nr_clusters = nr_clusters
    self.diss_func = diss_func
    self.linkage = linkage
    self.distance_threshold = distance_threshold

  def fit(self, X):
    # Initialize the clusters with each data point as a separate cluster
    clusters = [[x] for x in X]
    
    while len(clusters) > self.nr_clusters:
      # Compute the dissimilarity matrix for the current clusters
      diss_matrix = np.zeros((len(clusters), len(clusters)))
      for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
          diss_matrix[i, j] = self.diss_func(clusters[i], clusters[j])
          diss_matrix[j, i] = diss_matrix[i, j]
      
      # Find the indices of the two clusters to merge based on the chosen linkage method
      if self.linkage == 'single':
        merge_idx = np.unravel_index(np.argmin(diss_matrix), diss_matrix.shape)
      elif self.linkage == 'complete':
        merge_idx = np.unravel_index(np.argmax(diss_matrix), diss_matrix.shape)
      elif self.linkage == 'average':
        merge_idx = np.unravel_index(np.argmin(np.mean(diss_matrix, axis=0)), diss_matrix.shape)
      
      # Merge the clusters
      clusters[merge_idx[0]].extend(clusters[merge_idx[1]])
      del clusters[merge_idx[1]]
    
    self.clusters = clusters


class DBSCAN:
  def __init__(self, diss_func, epsilon=0.5, min_points=5):
    # epsilon is the maximum distance/dissimilarity between two points 
    # to be considered as in the neighborhood of each other
    # min_ponits is the number of points in a neighborhood for 
    # a point to be considered as a core point (a member of a cluster). 
    # This includes the point itself.
    # diss_func is the dissimilarity measure to compute the 
    # dissimilarity/distance between two data points 
    # YOUR CODE HERE
    pass

  def fit(self, X):
    # noise should be labeled as "-1" cluster
    # YOUR CODE HERE
    pass
