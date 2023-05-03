#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with a Neural Network mindset
# 
# Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize  cats. This assignment will step you through how to do this with a Neural Network mindset, and will also hone your intuitions about deep learning.
# 
# **Instructions:**
# - Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
# - Use `np.dot(X,Y)` to calculate dot products.
# 
# **You will learn to:**
# - Build the general architecture of a learning algorithm, including:
#     - Initializing parameters
#     - Calculating the cost function and its gradient
#     - Using an optimization algorithm (gradient descent) 
# - Gather all three functions above into a main model function, in the right order.
# 
# ## Important Note on Submission to the AutoGrader
# 
# Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:
# 
# 1. You have not added any _extra_ `print` statement(s) in the assignment.
# 2. You have not added any _extra_ code cell(s) in the assignment.
# 3. You have not changed any of the function parameters.
# 4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
# 5. You are not changing the assignment code where it is not required, like creating _extra_ variables.
# 
# If you do any of the following, you will get something like, `Grader not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/neural-networks-deep-learning/supplement/iLwon/h-ow-to-refresh-your-workspace).

# ## Table of Contents
# - [1 - Packages](#1)
# - [2 - Overview of the Problem set](#2)
#     - [Exercise 1](#ex-1)
#     - [Exercise 2](#ex-2)
# - [3 - General Architecture of the learning algorithm](#3)
# - [4 - Building the parts of our algorithm](#4)
#     - [4.1 - Helper functions](#4-1)
#         - [Exercise 3 - sigmoid](#ex-3)
#     - [4.2 - Initializing parameters](#4-2)
#         - [Exercise 4 - initialize_with_zeros](#ex-4)
#     - [4.3 - Forward and Backward propagation](#4-3)
#         - [Exercise 5 - propagate](#ex-5)
#     - [4.4 - Optimization](#4-4)
#         - [Exercise 6 - optimize](#ex-6)
#         - [Exercise 7 - predict](#ex-7)
# - [5 - Merge all functions into a model](#5)
#     - [Exercise 8 - model](#ex-8)
# - [6 - Further analysis (optional/ungraded exercise)](#6)
# - [7 - Test with your own image (optional/ungraded exercise)](#7)

# <a name='1'></a>
# ## 1 - Packages ##
# 
# First, let's run the cell below to import all the packages that you will need during this assignment. 
# - [numpy](https://numpy.org/doc/1.20/) is the fundamental package for scientific computing with Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
# - [PIL](https://pillow.readthedocs.io/en/stable/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.

# In[1]:


import numpy as n
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2p')


# <a name='2'></a>
# ## 2 - Overview of the Problem set ##
# 
# **Problem Statement**: You are given a dataset ("data.h5") containing:
#     - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#     - a test set of m_test images labeled as cat or non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
# 
# You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
# 
# Let's get more familiar with the dataset. Load the data by running the following code.

# In[2]:


# Loading the data (cat/non-cat)
df=load_dataset()
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).
# 
# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. Feel free also to change the `index` value and re-run to see other images. 

# In[3]:


# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs. 
# 
# <a name='ex-1'></a>
# ### Exercise 1
# Find the values for:
#     - m_train (number of training examples)
#     - m_test (number of test examples)
#     - num_px (= height = width of a training image)
# Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`.

# In[4]:


#(≈ 3 lines of code)
# m_train = 
# m_test = 
# num_px = 
# YOUR CODE STARTS HERE
m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px  = train_set_x_orig.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# **Expected Output for m_train, m_test and num_px**: 
# <table style="width:15%">
#   <tr>
#     <td> m_train </td>
#     <td> 209 </td> 
#   </tr>
#   
#   <tr>
#     <td>m_test</td>
#     <td> 50 </td> 
#   </tr>
#   
#   <tr>
#     <td>num_px</td>
#     <td> 64 </td> 
#   </tr>
#   
# </table>
# 

# For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px $*$ num_px $*$ 3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.
# 
# <a name='ex-2'></a>
# ### Exercise 2
# Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num\_px $*$ num\_px $*$ 3, 1).
# 
# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: 
# ```python
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
# ```

# In[5]:


# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
m_train,num_px1,num_px2,channel=train_set_x_orig.shape
img_size=num_px1*num_px2*channel
train_set_x_flatten=train_set_x_orig.reshape((m_train,img_size)).T
test_set_x_flatten= test_set_x_orig.reshape((test_set_x_orig.shape[0],img_size)).T
# Check that the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# **Expected Output**: 
# 
# <table style="width:35%">
#   <tr>
#     <td>train_set_x_flatten shape</td>
#     <td> (12288, 209)</td> 
#   </tr>
#   <tr>
#     <td>train_set_y shape</td>
#     <td>(1, 209)</td> 
#   </tr>
#   <tr>
#     <td>test_set_x_flatten shape</td>
#     <td>(12288, 50)</td> 
#   </tr>
#   <tr>
#     <td>test_set_y shape</td>
#     <td>(1, 50)</td> 
#   </tr>
# </table>

# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
# 
# One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
# 
# <!-- During the training of your model, you're going to multiply weights and add biases to some initial inputs in order to observe neuron activations. Then you backpropogate with the gradients to train the model. But, it is extremely important for each feature to have a similar range such that our gradients don't explode. You will see that more in detail later in the lectures. !--> 
# 
# Let's standardize our dataset.

# In[6]:


train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# <font color='blue'>
#     
#     
# **What you need to remember:**
# 
# Common steps for pre-processing a new dataset are:
# - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# - Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
# - "Standardize" the data

# <a name='3'></a>
# ## 3 - General Architecture of the learning algorithm ##
# 
# It's time to design a simple algorithm to distinguish cat images from non-cat images.
# 
# You will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**
# 
# <img src="images/LogReg_kiank.png" style="width:650px;height:400px;">
# 
# **Mathematical expression of the algorithm**:
# 
# For one example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
# 
# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
# 
# **Key steps**:
# In this exercise, you will carry out the following steps: 
#     - Initialize the parameters of the model
#     - Learn the parameters for the model by minimizing the cost  
#     - Use the learned parameters to make predictions (on the test set)
#     - Analyse the results and conclude

# <a name='4'></a>
# ## 4 - Building the parts of our algorithm ## 
# 
# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)
# 
# You often build 1-3 separately and integrate them into one function we call `model()`.
# 
# <a name='4-1'></a>
# ### 4.1 - Helper functions
# 
# <a name='ex-3'></a>
# ### Exercise 3 - sigmoid
# Using your code from "Python Basics", implement `sigmoid()`. As you've seen in the figure above, you need to compute $sigmoid(z) = \frac{1}{1 + e^{-z}}$ for $z = w^T x + b$ to make predictions. Use np.exp().

# In[7]:


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """
   
    s=1/ (1+np.exp(-z))
    
    return s


# In[8]:


print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

sigmoid_test(sigmoid)


# In[9]:


x = np.array([0.5, 0, 2.0])
output = sigmoid(x)
print(output)


# <a name='4-2'></a>
# ### 4.2 - Initializing parameters
# 
# <a name='ex-4'></a>
# ### Exercise 4 - initialize_with_zeros
# Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros. If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.

# In[40]:


# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = float(0)
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b



# In[41]:


dim = 2
w, b = initialize_with_zeros(dim)

assert type(b) == float
print ("w = " + str(w))
print ("b = " + str(b))

initialize_with_zeros_test_1(initialize_with_zeros)
initialize_with_zeros_test_2(initialize_with_zeros)


# <a name='4-3'></a>
# ### 4.3 - Forward and Backward propagation
# 
# Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
# 
# <a name='ex-5'></a>
# ### Exercise 5 - propagate
# Implement a function `propagate()` that computes the cost function and its gradient.
# 
# **Hints**:
# 
# Forward Propagation:
# - You get X
# - You compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
# - You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$
# 
# Here are the two formulas you will be using: 
# 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

# In[42]:


# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation
    """
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = - np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
    dw=np.dot(X,(A-Y).T)/m
    db=np.sum(A-Y)/m
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost


# In[43]:


w =  np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])
grads, cost = propagate(w, b, X, Y)

assert type(grads["dw"]) == np.ndarray
assert grads["dw"].shape == (2, 1)
assert type(grads["db"]) == np.float64


print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

propagate_test(propagate)


# **Expected output**
# 
# ```
# dw = [[ 0.25071532]
#  [-0.06604096]]
# db = -0.1250040450043965
# cost = 0.15900537707692405
# ```

# <a name='4-4'></a>
# ### 4.4 - Optimization
# - You have initialized your parameters.
# - You are also able to compute a cost function and its gradient.
# - Now, you want to update the parameters using gradient descent.
# 
# <a name='ex-6'></a>
# ### Exercise 6 - optimize
# Write down the optimization function. The goal is to learn $w$ and $b$ by minimizing the cost function $J$. For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.

# In[45]:


# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        grads,cost=propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


# In[46]:


params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("Costs = " + str(costs))

optimize_test(optimize)


# <a name='ex-7'></a>
# ### Exercise 7 - predict
# The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There are two steps to computing predictions:
# 
# 1. Calculate $\hat{Y} = A = \sigma(w^T X + b)$
# 
# 2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 

# In[47]:


# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# In[48]:


w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
print ("predictions = " + str(predict(w, b, X)))

predict_test(predict)


# <font color='blue'>
#     
# **What to remember:**
#     
# You've implemented several functions that:
# - Initialize (w,b)
# - Optimize the loss iteratively to learn parameters (w,b):
#     - Computing the cost and its gradient 
#     - Updating the parameters using gradient descent
# - Use the learned (w,b) to predict the labels for a given set of examples

# <a name='5'></a>
# ## 5 - Merge all functions into a model ##
# 
# You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.
# 
# <a name='ex-8'></a>
# ### Exercise 8 - model
# Implement the model function. Use the following notation:
#     - Y_prediction_test for your predictions on the test set
#     - Y_prediction_train for your predictions on the train set
#     - parameters, grads, costs for the outputs of optimize()

# In[49]:


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # (≈ 1 line of code)   
    # initialize parameters with zeros 
    # w, b = ...
    
    #(≈ 1 line of code)
    # Gradient descent 
    # params, grads, costs = ...
    
    # Retrieve parameters w and b from dictionary "params"
    # w = ...
    # b = ...
    
    # Predict test/train set examples (≈ 2 lines of code)
    # Y_prediction_test = ...
    # Y_prediction_train = ...
    
    # YOUR CODE STARTS HERE
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    # YOUR CODE ENDS HERE

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[50]:


from public_tests import *

model_test(model)


# If you pass all the tests, run the following cell to train your model.

# In[51]:


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)


# **Comment**: Training accuracy is close to 100%. This is a good sanity check: your model is working and has high enough capacity to fit the training data. Test accuracy is 70%. It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier. But no worries, you'll build an even better classifier next week!
# 
# Also, you see that the model is clearly overfitting the training data. Later in this specialization you will learn how to reduce overfitting, for example by using regularization. Using the code below (and changing the `index` variable) you can look at predictions on pictures of the test set.

# In[83]:


# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")


# Let's also plot the cost function and the gradients.

# In[52]:


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# **Interpretation**:
# You can see the cost decreasing. It shows that the parameters are being learned. However, you see that you could train the model even more on the training set. Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting. 

# <a name='6'></a>
# ## 6 - Further analysis (optional/ungraded exercise) ##
# 
# Congratulations on building your first image classification model. Let's analyze it further, and examine possible choices for the learning rate $\alpha$. 

# #### Choice of learning rate ####
# 
# **Reminder**:
# In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate $\alpha$  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.
# 
# Let's compare the learning curve of our model with several choices of learning rates. Run the cell below. This should take about 1 minute. Feel free also to try different values than the three we have initialized the `learning_rates` variable to contain, and see what happens. 

# In[ ]:


learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# **Interpretation**: 
# - Different learning rates give different costs and thus different predictions results.
# - If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost). 
# - A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
# - In deep learning, we usually recommend that you: 
#     - Choose the learning rate that better minimizes the cost function.
#     - If your model overfits, use other techniques to reduce overfitting. (We'll talk about this in later videos.) 
# 

# <a name='7'></a>
# ## 7 - Test with your own image (optional/ungraded exercise) ##
# 
# Congratulations on finishing this assignment. You can use your own image and see the output of your model. To do that:
#     1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
#     2. Add your image to this Jupyter Notebook's directory, in the "images" folder
#     3. Change your image's name in the following code
#     4. Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!

# In[ ]:


# change this to the name of your image file
my_image = "my_image.jpg"   

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


# <font color='blue'>
#     
# **What to remember from this assignment:**
# 1. Preprocessing the dataset is important.
# 2. You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
# 3. Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!

# Finally, if you'd like, we invite you to try different things on this Notebook. Make sure you submit before trying anything. Once you submit, things you can play with include:
#     - Play with the learning rate and the number of iterations
#     - Try different initialization methods and compare the results
#     - Test other preprocessings (center the data, or divide each row by its standard deviation)

# Bibliography:
# - http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# - https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
