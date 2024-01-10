import numpy as np

# you need Regresssion trees, so use either your implementation or sklearn's
#from decision_tree import RegressionTree 
from sklearn.tree import DecisionTreeRegressor

# Instead of using a Decision Tree with one level
# we can create another object for Decision Stump
# which will work faster since it will not compute impurity
# to decide on which feature to make a split

# after implementing this version, create a different Adaboost
# that uses decision trees with one level and check that it is 
# more inefficient compared to the below implementation.

class DecisionStump():
  def __init__(self):
    # we will use this attribute to convert the predictions
    # in case the error > 50%
    self.flip = 1
    # the feature index on which the split was made
    self.feature_index = None
    # the threshold based on which the split was made
    self.threshold = None
    # the confidence of the model (see the pseudocode from the lecture slides)
    self.alpha = None

class Adaboost():
  # this implementation supports only -1,1 label encoding
  def __init__(self, nr_estimators=5):
    # number of weak learners (Decision Stumps) to use
    self.nr_estimators = nr_estimators

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    nr_samples, nr_features = np.shape(X)

    # initialize the uniform weights for each training instance
    w = np.ones(nr_samples) / nr_samples
    
    self.models = []
    for i in range(self.nr_estimators):
        model = DecisionStump()

        # we set the initial error very high in order to select 
        # the model with lower error 
        min_error = 1 

        # we go over each feature as in case of decision tree
        # to decide which split leads to a smaller error
        # note that here we don't care about the impurity
        # even if we find a model with 90% error, we will flip the
        # sign of the predictions and will make it a model with 10% error
        for feature_id in range(nr_features):
          unique_values = np.unique(X[:, feature_id])
          thresholds = (unique_values[1:] + unique_values[:-1]) / 2
          for threshold in thresholds:
              # setting an intial value for the flip
              flip = 1
              # setting all the predictions as 1
              prediction = np.ones(nr_samples)
              # if the feature has values less than the fixed threshold
              # then it's prediction should be manually put as -1
              prediction[X[:, feature_id] < threshold] = -1

              # compute the weighted error (epsilon_t) for the resulting prediction
              error = sum(w[y != prediction])
              
              # if the model is worse than random guessing
              # then we need to set the flip variable to -1 
              # so that we can use it later, we also modify the error
              # accordingly
              if error > 0.5:
                error = 1 - error
                flip = -1

              # if this feature and threshold were the one giving 
              # the smallest error, then we store it's info in the 'model' object
              if error < min_error:
                model.flip = flip
                model.threshold = threshold
                model.feature_index = feature_id
                min_error = error
        
        # compute alpha based on the error of the 'best' decision stump
        model.alpha = 0.5 * np.log((1 - min_error)/(min_error + 1e-10))

        # obtain the predictions from the chosen decision stump
        # using the info stored in the 'model' object
        # don't forget about the flip if necessary
        # YOUR CODE HERE
        prediction = np.ones(nr_samples)
        negative_idx = (model.flip * X[:, model.feature_index] < model.flip * model.threshold)
        prediction[negative_idx] = -1


        # compute the weights and normalize them
        w *=np.exp(-model.alpha * y * prediction)
        w /= np.sum(w)

        # store the decision stump of the current iteration for later
        self.models.append(model)

  def predict(self, X):
    X = np.array(X)
    nr_samples = np.shape(X)[0]
    y_pred = np.zeros(nr_samples)

    # for each instance in X you should obtain the 'prediction'
    # from each decision stump (not forgetting about the flip variable)
    # then take the sum of 
    # all the individual predictions times their weights (alpha)
    # if the resulting amount is bigger than 0 then predict 1, otherwise -1
    # YOUR CODE HERE

    for model in self.models:
      prediction = np.ones(nr_samples)
      negative_idx = (model.flip * X[:, model.feature_index] < model.flip * model.threshold)
      prediction[negative_idx] = -1
      y_pred += model.alpha * prediction
    y_pred = np.sign(y_pred)
    return y_pred



class GradientBoosting:
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4, regression=False):      
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.regression = regression
    
    # write the square loss function as in the lectures
    def square_loss(y, y_pred): return -(y-y_pred)

    def cross_entropy(y, y_pred):
      y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
      return - (y / y_pred) + (1 - y) / (1 - y_pred)

    self.loss = square_loss
    self.loss_gradient = square_loss
    if not self.regression:
      self.loss_gradient = cross_entropy

  def fit(self, X, y):
    self.trees = [] # we will store the regression trees per iteration
    y_pred = np.full(np.shape(y), np.mean(y, axis=0))
    for i in range(self.n_estimators):     
      tree = DecisionTreeRegressor(
              min_samples_split=self.min_samples_split,
              min_impurity=self.min_impurity,
              max_depth=self.max_depth) # this is h(x) from our lectures
      residuals = self.loss_gradient(y, y_pred)
      tree.fit(X, residuals)
      h = tree.predict(X)
      y_pred -= self.learning_rate * h
      self.trees.append(tree) # stor the tree model

  def predict(self, X):
    # start with initial predictions as vector of 
    # the mean values of y_train (self.mean_y)
    y_pred = np.array([])
    # iterate over the regression trees and apply the same gradient updates
    # as in the fitting process, but using test instances
    for tree in self.trees:
      update = self.learning_rate * tree.predict(X)
      y_pred = -update if not y_pred.any() else y_pred - update

    if not self.regression:
      y_pred = 1/(1 + np.exp(-y_pred))
      y_pred = (y_pred >= 0.5) * 1
    
    return y_pred



class GradientBoostingRegressor(GradientBoosting):
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4):      
    
    super().__init__(
      n_estimators=n_estimators,
      learning_rate=learning_rate,
      min_samples_split=min_samples_split,
      min_impurity=min_impurity,
      max_depth=max_depth,
      regression=True
    )


class GradientBoostingClassifier(GradientBoosting):
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4):      

    super().__init__(n_estimators=n_estimators,learning_rate=learning_rate,min_samples_split=min_samples_split,min_impurity=min_impurity,max_depth=max_depth,regression=False)
