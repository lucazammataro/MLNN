# A Multi-Layer Neural Network Approach for Alzheimer’s Disease Detection in Magnetic Resonance Images
# Author: Luca Zammataro

import pandas as pd
import numpy as np 
from matplotlib.pyplot import imshow
import pickle


''' 
open the pickle file, and discover the contained dataframe
'''

with open('AugmentedAlzheimer.50X50.training.dataset.pickle', 'rb') as handle:
    ALZ = pickle.load(handle)


'''
transform each data frame's column into two distinct numpy objects, X, the training dataset, and y, the labels vector. 
(Transforming data into numpy objects is necessary for linear algebra calculations):
'''
X = np.stack(ALZ['X'])
y = np.int64(ALZ['y']) 

'''
Reshape a 2500-values vector extracted from one of the images
stored in the vector X (i.e.: #12848).
Use the NumPy method .reshape, specifiying the double argument '50'
then show the image with the function imshow, specifying the argument 
cmap='gray'
Running the code, image number #12848:
'''
image = X[12848].reshape(50, 50)
print('image:', 12848)
print('label:', y[12848])
imshow(image, cmap='gray')


'''
Instead of a one-by-one procedure, we prefer an easy function for randomly displaying fifty images picked from the dataset. Here is the code:
'''

'''
plotSamplesRandomly
Function for visualizing fifty randomly picked images, from the dataset
'''

def plotSamplesRandomly(X, y):
    
    from random import randint
    import matplotlib.pyplot as plt
    %matplotlib inline

    # create a list of randomly picked indexes.
    # the function randint creates the list, picking numbers in a 
    # range 0-33983, which is the length of X

    randomSelect = [randint(0, len(X)) for i in range(0, 51)]

    # reshape all the pictures on the n X n pixels, 
    # where n = sqrt(size of X), in this case 50 = sqrt(2500)
    w, h =int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))
    fig=plt.figure(figsize=(int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
  
    # Define a grid of 10 X 10 for the big plot. 
    columns = 10
    rows = 10

    # The for loop
    for i in range(1, 51):
  
        # create the 2-dimensional picture
        image = X[randomSelect[i]].reshape(w,h)
        ax = fig.add_subplot(rows, columns, i)

        # create a title for each pictures, containing #index and label
        title = "#"+str(randomSelect[i])+"; "+"y:"+str(y[randomSelect[i]])

        # set the title font size
        ax.set_title(title, fontsize=np.int(np.sqrt(X.shape[1])/2))                

        # don't display the axis
        ax.set_axis_off()
        
        # plot the image in grayscale
        plt.imshow(image, cmap='gray')
        
    # Show some sample randomly
    print('\nShow samples randomly:')
    plt.show()

'''
Run the plotSampleRandomly() function passing as arguments X and y:
'''
plotSamplesRandomly(X, y)



'''
Sigomid Logistic (Activation) Function
'''

def sigmoid(z): 
    g = 1.0 / (1.0 + np.exp(-z))
    return g
  


''' 
Plot the Sigmoid function, and its output:
'''

def plotSigmoid(z):
    import numpy as np
    from matplotlib import pyplot as plt

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.linspace(-10, 10, 100)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    plt.plot(x, sigmoid(x), color='blue')
    
    plt.plot(0, sigmoid(0), marker="o", markersize=5, 
             markeredgecolor="black", markerfacecolor="black")    
    if(z!=0):
        plt.text(0, sigmoid(0), "  0.5")
    
    plt.plot(z, sigmoid(z), marker="o", markersize=5,
             markeredgecolor="red", markerfacecolor="red")
    
    value = "{:5.3f}".format(sigmoid(z))
    plt.text(z, sigmoid(z), str("  z="+str(z)+"; value="+str(value)))

    
    plt.show()


'''
INITIALIZATION
'''
with open('AugmentedAlzheimer.50X50.training.dataset.pickle', 'rb') as handle:
    ALZ = pickle.load(handle) 
X = np.stack(ALZ['X'])
y = np.int64(ALZ['y'])     

'''
function "init"
Neural Network initialization and parameter setup
'''
def init(X, y):
    I  = X.shape[1]  # n x n MRI input size (2500)
    H1 = int(X.shape[1]*2/3) + max(set(y)) # hidden units size: 
                        # 2/3 of the input units + the number of output units (1670)
    O = max(set(y)) # output units size or labels (4)
    m = X.shape[0] 

    ini_Th1 = thetasRandomInit(I, H1)
    ini_Th2 = thetasRandomInit(H1, O)

    # Unroll parameters 
    ini_nn_params = np.concatenate((ini_Th1.T.flatten(), ini_Th2.T.flatten()))
    ini_nn_params = ini_nn_params.reshape(len(ini_nn_params), 1)

    print('\nNeural Network Parameters initialized!\n')
    print('\nNeural Network structure:\n')
    print('Input layer neurons:', I)
    print('1st hidden layer neurons:', H1)
    print('Output layer neurons:', O)
    
    return ini_nn_params, I, H1, O


'''
function "thetasRandomInit"
Random initialization of thetas
'''
def thetasRandomInit(L_in, L_out):
    e_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * e_init - e_init
    return W


'''
The W matrix, which contains a random values vector, is implemented based on an e_init value, set to 0.12 as default. 
Please copy and paste the previous code into a new cell and run it. For calling init and defining the NN parameters ini_nn_params, I, H1, O, type as follows:
'''
ini_nn_params, I, H1, O = init(X, y)


'''
TRAINING
'''

'''
function "training"
Neural Network training
'''
def training(ini_nn_params, I, H1, O, X, y):
    import scipy.optimize as opt

    lambd = 1
    print('\nTraining Neural Network... \n')


    result = opt.minimize(fun=nnCostFunction, 
                          x0=ini_nn_params,
                          args=(I, H1, O, X, y, lambd),
                          method='TNC', 
                          jac=True, 
                          options={'maxiter': 1000, "disp": True})

    params_trained = result.x

    Th1_trained = np.reshape(params_trained[0:(H1 * (I + 1)), ],
                                 (H1, I + 1))
    Th2_trained = np.reshape(params_trained[(H1 * (I + 1)):, ],
                                 (O, H1 + 1))

    print('\nTrained.\n')
    
    return Th1_trained, Th2_trained



'''
nnCostFunction
Implements the Neural Network Cost Function
'''
def nnCostFunction(nn_params, I, H1, O, X, y, lambd):
 
    # 1. RESHAPING
    # Reshape nn_params back into the parameters Th1 and Th2, 
    # the weight matrices

    Th1 = np.reshape(nn_params[0:(H1 * (I + 1)), ],
                         (H1, I + 1))
    Th2 = np.reshape(nn_params[(H1 * (I + 1)):, ],
                         (O, H1 + 1))

    # 2. SETUP OF Y
    # Setup the output (y) layout
    m = X.shape[0] 
    
    Y = np.zeros((m, O))
    for i in range(m):
        Y[i, y[i] - 1] = 1   
        
    # 3. INITIALIZE J, and THETAS
    J = 0
    Th1_grad = np.zeros(Th1.shape) 
    Th2_grad = np.zeros(Th2.shape) 
    
    # 4. PREPARE ALL THE VECTORS FOR THE FORWARD PROPAGATION
    # Six new vectors are generated here: a1, z2, a2, z3, a3, and h.
    # The vector a1 equals X (the input matrix), 
    # with a column of 1's added (bias units) as the first column.

    a1 = np.hstack((np.ones((m, 1)), X))  
    
    # z2 equals the product of a1 and Th1
    z2 = np.dot(a1, Th1.T) 

    # The vector a2 is created by adding a column of bias units 
    # after applying the sigmoid function to z2.
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2))) 
        
    # z3 equals the product of a2 and Th2
    z3 = np.dot(a2, Th2.T) 
    a3 = sigmoid(z3) 
    
    # The Hypotheses h is = a3
    h = a3 
    
    # 5. MAKE REGUKARIZED COST FUNCTION.
    # calculate P
    p = np.sum([
        np.sum(np.sum([np.power(Th1[:, 1:], 2)], 2)),
        np.sum(np.sum([np.power(Th2[:, 1:], 2)], 2))
    ])
    
    # Calculate Cost Function
    J = np.sum([
        np.divide(np.sum(np.sum([np.subtract(np.multiply((-Y), np.log(h)), 
                                             np.multiply((1-Y), np.log(1-h)))], 2)), m),
        np.divide(np.dot(lambd, p), np.dot(2, m))
    ])    


    # 6. FORWARD PROPAGATION
    # d3 is the difference between a3 and y. 
    d3 = np.subtract(a3, Y)
    
    z2 = np.hstack((np.ones((m, 1)), z2))
    d2 = d3.dot(Th2) * gradientDescent(z2)    
    d2 = d2[:, 1:]

    
    # GRADIENTS
    # Delta1 is the product of d2 and a1. 
    delta_1 = np.dot(d2.T, a1)

    # Delta2 is the product of d3 and a2. 
    delta_2 = np.dot(d3.T, a2)

    # Regularized Gradients.
    P1 = (lambd/m) * np.hstack([np.zeros((Th1.shape[0], 1)), Th1[:, 1:]])
    P2 = (lambd/m) * np.hstack([np.zeros((Th2.shape[0], 1)), Th2[:, 1:]])
    
    Theta_1_grad = (delta_1/m) + P1
    Theta_2_grad = (delta_2/m) + P2

    grad = np.hstack((Theta_1_grad.ravel(), Theta_2_grad.ravel()))
    
    
    
    return J, grad


'''
gradientDescent
returns the gradient of the sigmoid function

'''
def gradientDescent(z):
    g  = np.multiply(sigmoid(z), (1-sigmoid(z)))
    return g



'''
To run the training function, invoke it, specifying as arguments the ini_nn_params, I, H1, O, X, and, y:
'''
Th1_trained, Th2_trained = training(ini_nn_params, I, H1, O, X, y)


'''
After the download, you can upload the two files by typing:
'''

# Upload thetas from the two files
with open('ALZ.50.df_theta_1.pickle', 'rb') as handle:
        Th1_trained = pickle.load(handle).values
with open('ALZ.50.df_theta_2.pickle', 'rb') as handle:
        Th2_trained = pickle.load(handle).values 



'''
TESTING
'''

'''
testing()
'''
def testing(Th1, Th2, X, y):

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    a2 = sigmoid(X.dot(Th1.T))
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = sigmoid(a2.dot(Th2.T))

    pred = np.argmax(a3, axis=1)
    pred += 1  
    print('Training Set Accuracy:', np.mean(pred == y) * 100)
    return pred


'''
Before running the code, upload the testing version of the same dataset:
'''

# LOAD TESTING DATASET
with open('Alz.original.50X50.testing.dataset.pickle', 'rb') as handle:
    ALZ = pickle.load(handle) 
X = np.stack(ALZ['X'])
y = np.int64(ALZ['y'])     


'''
To run the testing, please invoke it, specifying as arguments the two trained θ vectors, X and Y:
'''

pred = testing(Th1_trained, Th2_trained, X, y)


'''
plotSamplesRandomlyWithPrediction
Function for visualizing fifty randomly picked images with their prediction.
'''

def plotSamplesRandomlyWithPrediction(X, y, pred):
    from random import randint
    from matplotlib import pyplot as plt
    
    # create a list of randomly picked indexes.
    # the function randint creates the list, picking numbers in a 
    # range 0-Xn, which is the length of X

    randomSelect = [randint(0, len(X)) for i in range(0, 51)]

    # reshape all the pictures on the n X n pixels, 
    w, h =int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))
    fig=plt.figure(figsize=(int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
  
    # Define a grid of 10 X 10 for the big plot. 
    columns = 10
    rows = 10

    # The for loop
    for i in range(1, 51):
  
        # create the 2-dimensional picture
        image = X[randomSelect[i]].reshape(w,h)
        ax = fig.add_subplot(rows, columns, i)

        # create a title for each pictures, containing #index and label
        #title = "#"+str(randomSelect[i])+"; "+"y="+str(y[randomSelect[i]])
        title = "#"+str(randomSelect[i])+"; "+"y:"+str(y[randomSelect[i]])+"; "+"p:"+str(pred[randomSelect[i]])

        # set the title font size
        ax.set_title(title, fontsize=np.int(np.sqrt(X.shape[1])/2))        

        # don't display the axis
        ax.set_axis_off()
        
        # plot the image in grayscale
        plt.imshow(image, cmap='gray')

    plt.show()


'''
And run the function as follows:
'''
plotSamplesRandomlyWithPrediction(X, y, pred)
