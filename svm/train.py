from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import numpy as np  
# loading the iris dataset 
iris = datasets.load_iris() 
  
# X -> features, y -> label 
X = np.genfromtxt(r'C:\Users\Amit Rai\Desktop\svm\train_X_svm.csv',delimiter=',',dtype=np.float64,skip_header=1)


y = np.genfromtxt(r'C:\Users\Amit Rai\Desktop\svm\train_Y_svm.csv',delimiter=',',dtype=np.float64)
print(X.shape,y.shape) 
  
# dividing X, y into train and test data 

  
# training a linear SVM classifier 
from sklearn.svm import SVC 

svm_model_linear = SVC(kernel = 'rbf', C = 100,gamma='auto').fit(X, y) 

        # creating a confusion matrix 
#cm = confusion_matrix(y_test, svm_predictions) 

import pickle

saved_model=pickle.dump(svm_model_linear, open('svm.sav', 'wb'))