#FOREST CLASSIFICATOR BASED ON RANDOM TREES ESTIMATORS TO CLASSIFY GALAXIES
#INTO SPIRAL, ELIPTICAL, MERGER GALAXIES 

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from support_functions import generate_features_targets, plot_confusion_matrix, calculate_accuracy 

#split data in training and testing sets
def splitdata_train_test(data, fraction_training):
  np.random.seed(0)
  np.random.shuffle(data)
  split = int(len(data) * fraction_training)
  training = data[:split]
  testing = data[split:]
  
  return training, testing

#return predictions and actual values 
def dtc_predict_actual(data):
  # split the data into training and testing sets using a training fraction of 0.7
  fraction_training = 0.7
  training, testing = splitdata_train_test(data, fraction_training)
  # generate the feature and targets for the training and test sets
  # i.e. train_features, train_targets, test_features, test_targets
  train_features, train_targets = generate_features_targets(training)
  test_features, test_targets = generate_features_targets(testing)
  # instantiate a decision tree classifier
  dtr = DecisionTreeClassifier()
  # train the classifier with the train_features and train_targets
  dtr.fit(train_features, train_targets)
  # get predictions for the test_features
  predictions = dtr.predict(test_features)
  # return the predictions and the test_targets
  return predictions, test_targets
  
#get predictions from a random forest classifier
def rf_predict_actual(data, n_estimators):
  # generate the features and targets
  features, targets = generate_features_targets(data)
  # instantiate a random forest classifier using n estimators
  rfc = RandomForestClassifier(n_estimators=n_estimators)
  # get predictions using 10-fold cross validation with cross_val_predict
  predicted = cross_val_predict(rfc, features, targets, cv=10)
  # return the predictions and their actual classes
  return predicted, targets
  
'''def calculate_accuracy(predicted_classes, actual_classes):
    return sum(actual_classes[:] == predicted_classes[:]) / float(len(actual_classes))'''

if __name__ == "__main__":
  data = np.load('Resources/galaxy_catalogue.npy')

  # set the fraction of data which should be in the training set
  fraction_training = 0.7

  # split the data
  training, testing = splitdata_train_test(data, fraction_training)

  # print the key values
  print('Number data galaxies:', len(data))
  print('Train fraction:', fraction_training)
  print('Number of galaxies in training set:', len(training))
  print('Number of galaxies in testing set:', len(testing))
  
  features, targets = generate_features_targets(data)

  # Print the shape of each array to check the arrays are the correct dimensions. 
  print("Features shape:", features.shape)
  print("Targets shape:", targets.shape)
  
  predicted_class, actual_class = dtc_predict_actual(data)

  # Print some of the initial results
  print("Some initial results...\n   predicted,  actual")
  for i in range(10):
    print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))
    
  #-------train the model to get predicted and actual classes---------------
  dtc = DecisionTreeClassifier()
  predicted = cross_val_predict(dtc, features, targets, cv=10)

  # calculate the model score using your function
  model_score = calculate_accuracy(predicted, targets)
  print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(targets))
  model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()
  
  #------------- get the predicted and actual classes-------------------
  number_estimators = 50              # Number of trees
  predicted, actual = rf_predict_actual(data, number_estimators)

  # calculate the model score using your function
  accuracy = calculate_accuracy(predicted, actual)
  print("Accuracy score:", accuracy)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(actual))
  model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

  # plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()