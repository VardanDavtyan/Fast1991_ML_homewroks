import numpy as np

# something useful for tracking algorithm's iterations
import progressbar

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

def get_bootstrap_samples(X, y, nr_bootstraps, nr_samples=None):
  # this function is for getting bootstrap samples with replacement 
  # from the initial dataset (X, y)
  # nr_bootstraps is the number of bootstraps needed
  # nr_samples is the number of data points to sample each time
  # it should be the size of X, if nr_samples is not provided
  # Hint: you may need np.random.choice function somewhere in this function
  #       bootstrap_sample_ids are the row indices that will be helpful for OOB score calculation
  if nr_samples is None:
    nr_samples = X.shape[0]

  bootstrap_samples = []
  bootstrap_sample_ids = []
  for i in range(nr_bootstraps):
    idx = np.random.choice(range(X.shape[0]), nr_samples, replace=True)
    bootstrap_samples.append([X[idx, :], y[idx]])
    bootstrap_sample_ids.append(idx)

  # YOUR CODE HERE
  return bootstrap_samples, bootstrap_sample_ids

class Bagging:
  def __init__(self, base_estimator, nr_estimators=10, oob_score=False):
    # number of models in the ensemble
    self.nr_estimators = nr_estimators
    
    # this can be any object that has 'fit', 'predict' methods
    self.base_estimator = base_estimator
    
    # whether we want to calculate the OOB score after training
    self.oob_score = oob_score
  
  def fit(self, X, y):
    # this method will fit a separate model (self.base_estimator)
    # on each bootstrap sample and each model should be stored
    # in order to use it in 'predict' method
  
    X = np.array(X)
    y = np.array(y)
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

    bootstrap_samples, bootstrap_sample_ids = get_bootstrap_samples(X, y,
                                              nr_bootstraps=self.nr_estimators)
    self.models = []
    for i in self.progressbar(range(self.nr_estimators)):
      model = self.base_estimator()
      X_boot, y_boot = bootstrap_samples[i]
      idx = bootstrap_samples[i]
      model.fit(X_boot, y_boot)
      model.idx = idx
      self.models.append(model)
    
    if self.oob_score:
      # this part is for calculating the OOB score
      oob_preds = []
      oob_ids = []
      for x_idx in range(len(X)):
        oob_preds_per_model = []
        counter  = 0
        for i in range(self.nr_estimators):
          if x_idx not in self.models[i].idx:
            counter += 1
            sample = X[x_idx].reshape(1, -1)
            pred = self.models[i].predict(sample)
            oob_preds_per_model.append(pred[0])
        
        if counter:
          oob_ids.append(x_idx)
          oob_preds.append(mode(oob_preds_per_model)[0][0])

      # YOUR CODE HERE
      self.oob_score = accuracy_score(y[np.array(oob_ids)], oob_preds)
  
  def predict(self, X):
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the ensemble
    # Hint: you may want to use 'mode' method from scipy.stats
    X = np.array(X)
    nr_estimators = self.nr_estimators
    y_preds = np.zeros((X.shape[0], nr_estimators))
    for i in range(nr_estimators):
      y_preds[:, i] = self.models[i].predict(X)
    return mode(y_preds, axis=1)[0]
  

class RandomForest:
  def __init__(self, nr_estimators=100, max_features=None, oob_score=False):
    # number of trees in the forest
    self.nr_estimators = nr_estimators   
    
    # this is the number of features to use for each tree
    # if not specified this should be set to sqrt(initial number of features) 
    self.max_features = max_features    
    
    self.oob_score = oob_score

  def fit(self, X, y):
    # this method will fit a separate tree
    # on each bootstrap sample and subset of features
    # each tree should be stored
    # in order to use it in 'predict' method
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

    X = np.array(X)
    y = np.array(y)
    nr_features = np.shape(X)[1]
    
    bootstrap_samples, bootstrap_sample_ids = get_bootstrap_samples(X, y, self.nr_estimators)
    
    if not self.max_features:
      self.max_features = int(np.sqrt(nr_features))
        
    
    self.trees = []
    for i in self.progressbar(range(self.nr_estimators)):
      tree = DecisionTreeClassifier()
      X_boot, y_boot = bootstrap_samples[i]
      row_idx = bootstrap_sample_ids[i]
      tree.row_idx = row_idx
      col_idx = np.random.choice(range(nr_features), size=self.max_features, replace=False)
      tree.feature_indices = col_idx
      tree.fit(X_boot[:, col_idx], y_boot)
      self.trees.append(tree)
    
    if self.oob_score:
      oob_preds = []
      oob_ids = []
      
      # Hint: here you should take into account that each tree is built using some of the features
      #       and you should keep only those features in the oob samples before giving to the trees 
      # YOUR CODE HERE 

      for x_idx in range(len(X)):
        oob_preds_per_model = []
        counter  = 0
        for i in range(self.nr_estimators):
          if x_idx not in self.trees[i].idx:
            counter += 1
            sample = X[x_idx].reshape(-1, 1)
            col_ids = self.trees[i].feature_indices
            pred = self.trees[i].predict(sample[:, col_ids])
            oob_preds_per_model.append(pred[0])        
        if counter:
          oob_ids.append(x_idx)
          oob_preds.append(mode(oob_preds_per_model)[0][0])
     
      self.oob_score = accuracy_score(y[np.array(oob_ids)], oob_preds)



  def predict(self, X):
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the forest
    # Hint: you may want to use 'mode' method from scipy.stats
    # besides the individual trees, you will also need the feature indices
    # it was trained on 
    y_preds = np.zeros((X.shape[0], self.nr_estimators))
    for i, tree in enumerate(self.trees):
      idx = tree.feature_indices
      y_preds[:, i] = tree.predict(X[:, idx])
    return mode(y_preds, axis=1)[0]
