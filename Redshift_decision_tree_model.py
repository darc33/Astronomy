#REDSHIFT USING SLOAN DIGITAL SKY SURVEY (SDSS)
#Decision Tree model for learning and get the distances of the galaxies

import numpy as np
import pydotplus as pydotplus
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from matplotlib import pyplot as plt

def get_features_targets(data):
  features = np.zeros((data.shape[0],4))
  targets = np.array(data['redshift'])
  features[:,0] = data['u'] - data['g']
  features[:,1] = data['g'] - data['r']
  features[:,2] = data['r'] - data['i']
  features[:,3] = data['i'] - data['z']
      
  return features,targets

#Return median differences between predicted and actual vals  
def median_diff(predicted, actual):
  med_diff = np.median(abs(predicted - actual))
  
  return med_diff
  
# split the data into training and testing features and predictions  
def split(features, targets):
    split = features.shape[0]//2
    train_features = features[:split]
    test_features = features[split:]
    train_targets = targets[:split]
    test_targets = targets[split:]
    
    return train_features, test_features, train_targets, test_targets
    
  
#split features and targets then validate the model
def validate_model(model, features, targets):
 
  train_features, test_features, train_targets, test_targets = split(features, targets)
  # train the model
  model.fit(train_features, train_targets)
  # get the predicted_redshifts
  predictions = model.predict(test_features)
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)
  
#Plot the decision tree but it break the memory of PC  
def plot_tree(model, features, targets):
    model.fit(features, targets)

    dot_data = export_graphviz(model, out_file=None,feature_names=['u - g', 'g - r', 'r - i', 'i - z'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_jpg("decision_tree.jpg")
    
    return
    
#Plot the redshift of the data    
def plot_redshift(data):
    plt.figure()
    data = data[data['redshift']<=3.490382]    
    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    u_g = data['u'] - data['g']
    r_i = data['r'] - data['i']
    # Make a redshift array
    redshift = np.array(data['redshift'])        
    # Create the plot with plt.scatter and plt.colorbar
    plt.scatter(u_g, r_i, s=5, c=redshift, cmap = cmap, lw=0)
    cbar = plt.colorbar()
    # Define your axis labels and plot title
    plt.xlabel('Colour index u-g')
    plt.ylabel('Colour index r-i')
    plt.title('Redshift (colour) u-g versus r-i')
    cbar.set_label('Redshift')
    # Set any axis limits
    axes = plt.gca()
    axes.set_xlim([-0.5,2.5])
    axes.set_ylim([-0.5,0.8])
    plt.show()
    
    return
    
#Proves different depths to find out the less error depth
def accuracy_by_treedepth(features, targets, depths):
  # split the data into testing and training sets
  train_features, test_features, train_targets, test_targets = split(features, targets)
  # initialise arrays or lists to store the accuracies for the below loop
  acc_train = []
  acc_test = []
  # loop through depths
  for depth in depths:
    # initialize model with the maximum depth. 
    dtr = DecisionTreeRegressor(max_depth=depth)

    # train the model using the training set
    dtr.fit(train_features, train_targets)

    # get the predictions for the training set and calculate their median_diff
    predictions_trn = dtr.predict(train_features)
    acc_train.append(median_diff(train_targets, predictions_trn))
    # get the predictions for the testing set and calculate their median_diff
    predictions_test = dtr.predict(test_features)
    acc_test.append(median_diff(test_targets, predictions_test))    
  # return the accuracies for the training and testing sets
  return acc_train, acc_test  

#With a fixed depth test different splits of the train and test sets of the data    
def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  acc = []
  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)
    # predict using the model
    predictions = model.predict(test_features)  
    # calculate the median_diff from predicted values and append to results array
    acc.append(median_diff(test_targets, predictions))
 
  # return the list with your median difference values
  return acc  

#Return all the predictions for a different splits of the train and test sets of the data    
def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    # fit the model for the current set
    model.fit(train_features, train_targets)    
    # predict using the model
    predictions = model.predict(test_features)    
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions 

#split the data set into galaxies and QSO  
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  galaxies = data[data['spec_class'] == b'GALAXY']
  QSO = data[data['spec_class'] == b'QSO']
  # return the seperated galaxies and qsos arrays
  return galaxies,QSO

#Get the mean of the median of the error between test and train sets for galaxies or QSO  
def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10)) 

if __name__ == "__main__":
  plt.close("all")
  # load the data
  data = np.load('Resources/sdss_galaxy_colors.npy')
   
  # call our function 
  features, targets = get_features_targets(data)
    
  # print the shape of the returned arrays
  print('Features: ', features[:2])
  print('Targets: ', targets[:2])
  
  # initialize model
  dtr = DecisionTreeRegressor(max_depth=19)
  
  # train the model
  dtr.fit(features, targets)
  
  # make predictions using the same features
  predictions = dtr.predict(features)
  
  # print out the first 4 predicted redshifts
  print('Predictions: ', predictions[:4])
  
  # call your function to measure the accuracy of the predictions
  diff = median_diff(predictions, targets)

  # print the median difference
  print("Median difference: {:0.3f}".format(diff))
  
  # validate the model and print the med_diff
  diff = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))
    
  plot_redshift(data)
  
  # Generate several depths to test
  tree_depths = [i for i in range(1, 36, 2)]

  # Call the function
  train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
  print("Depth with lowest median difference : {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))
    
  # Plot the results
  plt.figure()
  train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
  test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
  plt.xlabel("Maximum Tree Depth")
  plt.ylabel("Median of Differences")
  plt.legend()
  plt.show()
  
  # call your cross validation function
  diffs = cross_validate_model(dtr, features, targets, 10)

  # Print the values
  print('Differences: {}'.format(', '.join(['{:.3f}'.format(val) for val in diffs])))
  print('Mean difference: {:.3f}'.format(np.mean(diffs)))
  
  # call your cross validation function
  predictions = cross_validate_predictions(dtr, features, targets, 10)

  # calculate and print the rmsd as a sanity check
  diffs = median_diff(predictions, targets)
  print('Median difference: {:.3f}'.format(diffs))

  # plot the results to see how well our model looks
  plt.figure()
  plt.scatter(targets, predictions, s=0.4)
  plt.xlim((0, targets.max()))
  plt.ylim((0, predictions.max()))
  plt.xlabel('Measured Redshift')
  plt.ylabel('Predicted Redshift')
  plt.show()
  
  # Split the data set into galaxies and QSOs
  galaxies, qsos= split_galaxies_qsos(data)
  
  # Here we cross validate the model and get the cross-validated median difference
  # The cross_validated_med_diff function is in "written_functions"
  galaxy_med_diff = cross_validate_median_diff(galaxies)
  qso_med_diff = cross_validate_median_diff(qsos)

  # Print the results
  print("Median difference for Galaxies: {:.3f}".format(galaxy_med_diff))
  print("Median difference for QSOs: {:.3f}".format(qso_med_diff))
    