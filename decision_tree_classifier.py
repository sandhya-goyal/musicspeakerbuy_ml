# Decision Tree Classifier

# Step-1) Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step-2) Importing the Dataset

Dataset =  pd.read_csv('Alexa_dataset.csv') 
X = Dataset.iloc[:, [2, 3]].values
y = Dataset.iloc[:, 4].values

# Step-3) Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Step-4) Feature Scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

# Decision Tree Classifier

# Step-5) Fitting Decision Tree Classification to the Training set 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier( criterion =  'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Step-6) Predicting the Decision Tree Classifier
y_pred = classifier.predict(X_test)

# Step-7) Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Step-8) Performance metrics

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is %f' % accuracy)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is %f' % precision)
 
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is %f' % recall)

# plotting the graph

"""1.8: Function definition for visualization of results"""

# Function definition for visualization of results
def Visualizer(argument1, arguement2):
  from matplotlib.colors import ListedColormap
  X_set, y_set = argument1, arguement2
  X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop        
  =X_set[:,0].max()+1, step=0.01), 
  np.arange(start=X_set[:, 1].min() - 1, stop = 
  X_set[:, 1].max() + 1, step = 0.01))
  plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), 
  X2.ravel()]).T).reshape(X1.shape),
  alpha= 0.75, cmap = ListedColormap(('red', 'green')))

  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label=j)
  plt.xlabel('Age')
  plt.ylabel('Minutes_of_Music_Consumed')
  plt.legend()
  plt.show()

"""1.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""1.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)

