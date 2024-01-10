import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with hamming distance """

  def __init__(self, k=1):
    """
    Initializing the KNN object

    Inputs:
    - k: The number of nearest neighbors.
    """
    self.k = k
    
  def fit(self, X_train, y_train):
    """
    This method fits the data, which means
    memorizing the training data. 
    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D) 
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
    self.X_train = X_train
    self.y_train = y_train

  
  def predict(self, X_test): 
    """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just 
    memorizing the training data. 
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) 

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists, k=self.k)


  def compute_distances(self, X_test):
    """
    Compute the hamming distance between each test point in X_test and each training point
    in self.X_train.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the hamming distance between the ith test point and the jth training
      point.
    """
    ##### YOUR CODE STARTS HERE ##### 
    #################################
    X_train = self.X_train
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i, :] = (X_train != X_test[i,:]).sum(axis=1)
    ###############################
    ##### YOUR CODE ENDS HERE #####          
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data  
    """
    ##### YOUR CODE STARTS HERE ##### 
    #################################
    from scipy import stats
    def majority_class(y): return stats.mode(y)[0][0]

    if k == 1:
      y_pred = self.y_train[dists.argmin(axis=1)]
    else:
      y_pred = []
      for i in range(dists.shape[0]):
        closest_k = np.argsort(dists[i, :])[:k]
        closest_y = self.y_train[closest_k]
        y_pred.append(majority_class(closest_y))
    ###############################
    ##### YOUR CODE ENDS HERE #####              
    return y_pred
