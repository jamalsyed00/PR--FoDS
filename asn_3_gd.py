import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import random


"""data_preprocessing_"""

df = pd.read_csv("insurance.txt")
df.drop(['children'],axis = 1,inplace = True)

x = np.array(df.drop(['charges'], axis = 1))

polynom_feat = []
for j in range(1,11):
  poly = PolynomialFeatures(j)
  data = poly.fit_transform(x)
  polynom_feat.append(np.array(data))

def standardize(num, mu, sd):
    return (num - mu) / sd



def preprocessing_data(df):
  for j,feat in enumerate(df.T):
      mean = feat.mean()
      std = feat.std()
      feat = standardize(feat,mean,std)
      df[:,j] = feat
  df[:, 0] =  1    
  return df

for k in range(10):
  polynom_feat[k] = preprocessing_data(polynom_feat[k])

#target attribute (standardizing)
mean_y = df.mean()['charges']
sd_y = df.std()['charges']
df['charges']  = df['charges'].apply(lambda num : standardize(num,mean_y,sd_y)) 
Y = np.array((df['charges'])).reshape(len(df),1)

train_ratio = int(0.7*len(df))
valid_ratio = int(0.2*len(df))
train_Y = Y[:train_ratio]
val_Y =  Y[train_ratio:train_ratio+valid_ratio]
test_Y = Y[train_ratio+valid_ratio:]



"""FUNCTIONS"""

def grad_des(w,X,y,alpha,error_print,regre,penalty):
  err = []
  for i in range(10000):
    if error_print == True:
      if i%1000 == 0:
        print(cost_function(w,X,y))
    if i%50 == 0:
      err.append(cost_function(w,X,y))
    C = X.transpose()
    D = X.dot(w) - y
    if regre == "noreg":
      dw = C.dot(D) 
    elif regre == "lasso":
      dw = C.dot(D) + penalty*np.sign(w)
      dw[0] = dw[0] - penalty*np.sign(w[0])
    elif regre == "ridge":
      dw = C.dot(D) + penalty*2*w
      dw[0] = dw[0] - penalty*2*w[0]
    w = w - (alpha*dw)/len(X)
  return w,err

def cost_function(theta,x,y):
  return (np.sum((np.dot(x,theta) - y)**2))/2  

def predict(theta,x):
  return np.dot(x,theta)

def RMSE(y_obs,y_pred):
  return np.sqrt(((y_obs - y_pred) ** 2).mean())

"""POLYNOMIAL REGRESSION WITHOUT REGULARIZATION"""

param_list_without_reg = []
errors_train = []
errors_val = []
test_errors = []
rmse_train = []
rmse_val = []
rmse_test = []

for i in range(10):
  w = np.zeros((polynom_feat[i].shape[1],1))
  train_X = polynom_feat[i][:train_ratio]
  val_X = polynom_feat[i][train_ratio:train_ratio+valid_ratio]
  test_X = polynom_feat[i][train_ratio + valid_ratio:]
  print("Degree of Polynomial {}".format(i+1))
  print("Cost Function after 1000 epochs: ")
  weights,error = grad_des(w,train_X,train_Y,0.01,error_print = True,regre = "noreg",penalty = 0)
  errors_train.append(cost_function(weights,train_X,train_Y))
  errors_val.append(cost_function(weights,val_X,val_Y))
  test_errors.append(cost_function(weights,test_X,test_Y))

  pred_train = predict(weights,train_X)
  pred_val = predict(weights,val_X)
  pred_test = predict(weights,test_X)

  rmse_train.append(RMSE(train_Y,pred_train))
  rmse_val.append(RMSE(val_Y,pred_val))
  rmse_test.append(RMSE(test_Y,pred_test))
  print("")
  param_list_without_reg.append(weights)

for i in range(10):
  print("Model of Degree {}:".format(i+1))
  print("Training data SSE = {}".format(errors_train[i]))
  print("Validation data SSE = {}".format(errors_val[i]))
  print("Testing data SSE = {}".format(test_errors[i]))
  print("Training data RMSE value = {}".format(rmse_train[i]))
  print("Validation data RMSE value = {}".format(rmse_val[i]))
  print("Testing data RMSE value = {}".format(rmse_test[i]))
  print(" ")

"""VISUALIZATION (OVERFITTING)"""

x = [1,2,3,4,5,6,7,8,9,10]
plt.subplot()
plt.plot(x,errors_train)
plt.xlabel("Degree of Polynomial",fontsize = 13)
plt.ylabel("Training Error",fontsize = 13)
plt.title("Training Error v Degree of Polynomial",fontsize = 15)
plt.show()
plt.subplot()
plt.plot(x,errors_val)
plt.xlabel("Degree of Polynomial",fontsize = 13)
plt.ylabel("Validation Error",fontsize = 13)
plt.title("Validation Error v Degree of Polynomial",fontsize = 15)
plt.show()
plt.subplot()
plt.plot(x,test_errors)
plt.xlabel("Degree of Polynomial",fontsize = 13)
plt.ylabel("Testing error",fontsize = 13)
plt.title("Testing Error v Degree of Polynomial",fontsize = 15)
plt.show()


hyper = [0.043427172, 0.08096094, 0.15857641, 0.158742, 0.26073880,0.38243637, 0.39925635, 0.4476713 , 0.83824543, 0.9628143]

"""LASSO REGULARIZATION"""

opt_hyper = []
for i in range(10):
  val_error = []
  for pen in hyper:
    w = np.zeros((polynom_feat[i].shape[1],1))
    train_X = polynom_feat[i][:train_ratio]
    val_X = polynom_feat[i][train_ratio:train_ratio+valid_ratio]
    weights,error = grad_des(w,train_X,train_Y,0.01,error_print = False,regre = "lasso",penalty = pen)
    print("Validation data Cost Function of degree {} and hyperparameter {} is {}".format(i+1,pen,cost_function(weights,val_X,val_Y)))
    val_error.append(cost_function(weights,val_X,val_Y))
    print("")
  val_error = np.array(val_error)  
  ind = np.argmin(val_error)
  opt_hyper.append(hyper[ind])

"""POLYNOMIAL REGRESSION (LASSO REGULARIZATION) OPTIMAL HYPERPARAMETERS"""

param_list_lasso = []
errors_train = []
errors_val = []
test_errors = []

for k in range(10):
    w = np.zeros((polynom_feat[k].shape[1],1))
    train_X = polynom_feat[k][:train_ratio]
    val_X = polynom_feat[k][train_ratio:train_ratio+valid_ratio]
    test_X = polynom_feat[k][train_ratio + valid_ratio:]
    print("Lasso regularization: degree= {} and lambda= {}".format(k+1,opt_hyper[k]))
    print("Cost Function every 1000 iterations:")
    weights,error = grad_des(w,train_X,train_Y,0.01,error_print = True,regre = "lasso",penalty = opt_hyper[k])

    errors_train.append(cost_function(weights,train_X,train_Y))
    errors_val.append(cost_function(weights,val_X,val_Y))
    test_errors.append(cost_function(weights,test_X,test_Y))

    pred_train = predict(weights,train_X)
    pred_val = predict(weights,val_X)
    pred_test = predict(weights,test_X)
    print("")
    param_list_lasso.append(weights)

for i in range(10):
    print("Model of Degree {} AND LAMBDA {}:".format(i+1,opt_hyper[i]))
    print("Training data SSE = {}".format(errors_train[i]))
    print("Validation data SSE = {}".format(errors_val[i]))
    print("Testing data SSE = {}".format(test_errors[i]))
    print(" ")

"""(RIDGE REGULARIZATION) Finding OPTIMAL HYPERPARAMETERS"""

opt_hyper = []
for i in range(10):
  val_error = []
  for pen in hyper:
    w = np.zeros((polynom_feat[i].shape[1],1))
    train_X = polynom_feat[i][:train_ratio]
    val_X = polynom_feat[i][train_ratio:train_ratio+valid_ratio]
    weights,error = grad_des(w,train_X,train_Y,0.01,error_print = False,regre = "ridge",penalty = pen)
    print("Validation data Cost Function of degree {} and hyperparameter {} is {}".format(i+1,pen,cost_function(weights,val_X,val_Y)))
    val_error.append(cost_function(weights,val_X,val_Y))
    print("")
  val_error = np.array(val_error)  
  ind = np.argmin(val_error)
  opt_hyper.append(hyper[ind])

"""RIDGE REGULARIZATION (OPTIMAL HYPERPARAMETERS)"""

param_list_ridge = []
errors_train = []
errors_val = []
test_errors = []

for i in range(10):
    w = np.zeros((polynom_feat[i].shape[1],1))
    train_X = polynom_feat[i][:train_ratio]
    val_X = polynom_feat[i][train_ratio:train_ratio+valid_ratio]
    test_X = polynom_feat[i][train_ratio + valid_ratio:]
    print("Ridge regularization degree= {} and lambda= {}".format(i+1,opt_hyper[i]))
    print("Cost Function every 1000 iterations:")
    weights,error = grad_des(w,train_X,train_Y,0.01,error_print = True,regre = "ridge",penalty = opt_hyper[i])

    errors_train.append(cost_function(weights,train_X,train_Y))
    errors_val.append(cost_function(weights,val_X,val_Y))
    test_errors.append(cost_function(weights,test_X,test_Y))

    pred_train = predict(weights,train_X)
    pred_val = predict(weights,val_X)
    pred_test = predict(weights,test_X)

    print("")
    param_list_ridge.append(weights)

for i in range(10):
    print("Model of Degree {} AND LAMBDA {}:".format(i+1,opt_hyper[i]))
    print("Training data SSE = {}".format(errors_train[i]))
    print("Validation data SSE = {}".format(errors_val[i]))
    print("Testing data SSE = {}".format(test_errors[i]))
    print(" ")

"""degree 4 weights"""

print("Degree 4 => without regularization WEIGHTS {}".format(param_list_without_reg[3]))
print("Degree 4 => Lasso regularization WEIGHTS {}".format(param_list_lasso[3]))
print("Degree 4 => Ridge regularization WEIGHTS {}".format(param_list_ridge[3]))

"""3D PLOTS"""

for i in range(10):
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure(figsize=(12,9))
  ax = fig.add_subplot(111, projection='3d')
  X = train_X[:][:,1]
  Y = train_X[:][:,2]
  X, Y = np.meshgrid(X, Y)
  X = X.flatten()
  Y = Y.flatten()
  matr = np.array(np.vstack((X, Y)).T,dtype = 'float')
  poly = PolynomialFeatures(i+1)
  data = poly.fit_transform(matr)
  z = predict(param_list_without_reg[i],data)
  ax.set_title('Degree of Polynomial {}'.format(i+1),fontsize=18)
  ax.set_xlabel('Age', fontsize=17,y=5)
  ax.set_ylabel('BMI',fontsize = 17,y=5)
  ax.set_zlabel('Charges', fontsize=17,y=5)
  my_cmap = plt.get_cmap('hot') 
  ax.scatter(train_X[:][:,1],train_X[:][:,2],train_Y, zdir='z', s=20, c=None, depthshade=True,cmap = my_cmap)
  trisurf = ax.plot_trisurf(X, Y, z.flatten(), cmap = my_cmap,linewidth = 0.2,antialiased = True,edgecolor = 'grey') 
  plt.show()
