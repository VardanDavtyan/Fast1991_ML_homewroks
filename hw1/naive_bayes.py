import numpy as np

class NaiveBayes:
  def __init__(self, smoothing=False):
      # initialize Laplace smoothing parameter
      self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
      # use this method to learn the model
      # if you feel it is easier to calculate priors 
      # and likelihoods at the same time
      # then feel free to change this method
      self.X_train = X_train
      self.y_train = y_train
      self.labels = np.unique(self.y_train)
      self.priors = self.calculate_priors()
      self.likelihoods = self.calculate_likelihoods()      
      
  def predict(self, X_test):
      # recall: posterior is P(label_i|feature_j)
      # hint: Posterior probability is a matrix of size 
      #       m*n (m samples and n labels)
      #       our prediction for each instance in data is the class that 
      #       has the highest posterior probability. 
      #       You do not need to normalize your posterior, 
      #       meaning that for classification, prior and likelihood are enough
      #       and there is no need to divide by evidence. Think why!
      # return: nd array of class labels (predicted)
      ##### YOUR CODE STARTS HERE ##### 
      predictions = []
      for i in range(X_test.shape[0]):
        ##### YOUR CODE HERE #####
        pred = []
        for l in range(self.labels.shape[0]):
          pred.append(self.priors[l])
          for j in range(X_test.shape[1]):
            pred[l] *= self.likelihoods[self.labels[l]][j][X_test[i, j]]
            
        predictions.append(self.labels[np.argmax(pred)])

      return np.array(predictions)

  def calculate_priors(self):
      # recall: prior is P(label=l_i)
      # hint: store priors in a pandas Series or a list
      ##### YOUR CODE STARTS HERE #####             
      
      priors = []
      ##### YOUR CODE HERE #####
      for i in range(self.labels.shape[0]):
        count = np.sum(self.y_train == self.labels[i])
        priors.append(count/len(self.y_train))

      return np.array(priors)
  
  def calculate_likelihoods(self):
      # recall: likelihood is P(feature=f_j|label=l_i)
      # hint: store likelihoods in a data structure like dictionary:
      #        feature_j = [likelihood_k]
      #        likelihoods = {label_i: [feature_j]}
      #       Where j implies iteration over features, and 
      #             k implies iteration over different values of feature j. 
      #       Also, i implies iteration over different values of label. 
      #       Likelihoods, is then a dictionary that maps different label 
      #       values to its corresponding likelihoods with respect to feature
      #       values (list of lists).
      #
      #       NB: The above pseudocode is for the purpose of understanding
      #           the logic, but it could also be implemented as it is.
      #           You are free to use any other data structure 
      #           or way that is convenient to you!
      #
      #       More Coding Hints: You are encouraged to use Numpy/Pandas as much as
      #       possible for all these parts as it comes with flexible and
      #       convenient indexing features which makes the task easier.
      ##### YOUR CODE STARTS HERE ##### 
      likelihoods = {}
      ##### YOUR CODE HERE #####
      for l in self.labels:
        likelihoods[l] = []
        for i in range(self.X_train.shape[1]):
          values = np.unique(self.X_train[:, i])
          likelihood = {}
          for v in values:
            likelihood[v] = (np.sum(self.X_train[self.y_train == l, i] == v) +
                             self.smoothing)/(np.sum(self.y_train == l) +
                                              values.shape[0]*self.smoothing)

          likelihoods[l].append(likelihood)

      return likelihoods

