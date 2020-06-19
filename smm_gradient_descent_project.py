#!/usr/bin/env python
# coding: utf-8

# # SMM with Gradient Descent

# ## Python Imports

# In[1]:


import numpy as np
import random
import copy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from argparse import ArgumentParser


#get_ipython().run_line_magic('matplotlib', 'inline')
parser = ArgumentParser(description="Stabilisation matrix method with gradient descent")
parser.add_argument("-l", action="store", dest="lamb", type=float, default=0.01, help="Lambda (default: 0.01)")
parser.add_argument("-t", action="store", dest="training_file", type=str, help="File with training data")
parser.add_argument("-e", action="store", dest="evaluation_file", type=str, help="File with evaluation data for early stopping")
parser.add_argument("-epi", action="store", dest="epsilon", type=float,default=0.05, help="Epsilon (default 0.05)")
parser.add_argument("-s", action="store", dest="seed",  type=int, default=1, help="Seed for random numbers (default 1)")
parser.add_argument("-i", action="store", dest="epochs",  type=int, default=100, help="Number of epochs to train (default 100)")
parser.add_argument("-v", action="store", dest="validation_file", type=str, help="File with evaluation data for cross-validation")
parser.add_argument("-bl", action="store_true", dest="blosum_scheme", help=" Use Blosum encoding (default sparse encoding)")
parser.add_argument("-data", action="store", dest="data_dir", type=str, default="./data/", help="File with data, i.e. the matrix/ folder (default './data/')")


args = parser.parse_args()
lamb = args.lamb
training_file = args.training_file
evaluation_file=args.evaluation_file
epsilon = args.epsilon
seed=args.seed
epochs=args.epochs
validation_file=args.validation_file
blosum_scheme=args.blosum_scheme
data_dir=args.data_dir
# ## Data Imports

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[2]:




# ### Training Data

# In[3]:


#training_file = data_dir + "SMM/B4001/f000"
#training_file = data_dir + "SMM/A2403_training"

training = np.loadtxt(training_file, dtype=str)


# ### Evaluation Data

# In[4]:


#evaluation_file = data_dir + "SMM/B4001/c000"
#evaluation_file = data_dir + "SMM/A2403_evaluation"
evaluation = np.loadtxt(evaluation_file, dtype=str)

# ### Validation Data
validation = np.loadtxt(validation_file, dtype=str)


# ### Alphabet

# In[5]:


alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)


# ### Sparse Encoding Scheme

# In[6]:


sparse_file = data_dir + "Matrices/sparse"    # 20 x 20 with 1's thorugh the diagonal
_sparse = np.loadtxt(sparse_file, dtype=float)
sparse = {}

for i, letter_1 in enumerate(alphabet):
    
    sparse[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        sparse[letter_1][letter_2] = _sparse[i, j]    # the same as sparse file but with peptide letters as rows and columns


# ### Blosum50 Encoding Scheme

# In[7]:


blosum_file = data_dir + "Matrices/blosum50"

_blosum50 = np.loadtxt(blosum_file, dtype=float).reshape((24, -1)).T

blosum50 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum50[letter_1][letter_2] = _blosum50[i, j] / 5.0


# ## Peptide Encoding

# In[8]:


def encode(peptides, encoding_scheme, alphabet):   
    
    encoded_peptides = []

    for peptide in peptides:

        encoded_peptide = []

        for peptide_letter in peptide:

            for alphabet_letter in alphabet:

                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])    #note the peptide is first, then alphabet

        encoded_peptides.append(encoded_peptide)
        
    return np.array(encoded_peptides)

if blosum_scheme:
    scheme = blosum50
else:
    scheme = sparse
	
# ## Error Function

# In[9]:


def cumulative_error(peptides, y, lamb, weights):

    error = 0
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]

        # get target prediction value
        y_target = y[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
            
        # calculate error
        error += 1.0/2 * (y_pred - y_target)**2
        
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
        
    return gerror, error


# ## Predict value for a peptide list

# In[10]:


def predict(peptides, weights):

    pred = []
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
        
        pred.append(y_pred)
        
    return pred


# ## Calculate MSE between two vectors

# In[11]:


def cal_mse(vec1, vec2):
    
    mse = 0
    
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
        
    mse /= len(vec1)
    
    return( mse)


# ## Gradient Descent

# In[12]:


def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    
    # do is dE/dO
    #do = XX
    do = y_pred - y_target
    
    for i in range(0, len(weights)):
        
        #de_dw_i = XX
        de_dw_i = do * peptide[i] + 2 * lamb_N * weights[i]
        #weights[i] -= XX
        weights[i] -= epsilon * de_dw_i


# ## Main Loop
# 
# 

# In[13]:


# Random seed 
np.random.seed( 1 )

# early stopping
early_stopping = True
best_error = np.inf

# peptides
peptides = training[:, 0]
peptides = encode(peptides, scheme, alphabet)
N = len(peptides)

# target values
y = np.array(training[:, 1], dtype=float)

#evaluation peptides
evaluation_peptides = evaluation[:, 0]
evaluation_peptides = encode(evaluation_peptides, scheme, alphabet)

#evaluation targets
evaluation_targets = np.array(evaluation[:, 1], dtype=float)

# weights
input_dim  = len(peptides[0])
output_dim = 1
w_bound = 0.1
weights = np.random.uniform(-w_bound, w_bound, size=input_dim)
best_weights = np.zeros(input_dim)
# training epochs
#epochs = 100

# regularization lambda
#lamb = 1
#lamb = 10
#lamb = 0.01

# regularization lambda per target value
lamb_N = lamb/N

# learning rate
#epsilon = 0.05

# error  plot
gerror_plot = []
mse_plot = []
train_mse_plot = []
eval_mse_plot = []
train_pcc_plot = []
eval_pcc_plot = []

# for each training epoch
for e in range(0, epochs):

    # for each peptide
    for i in range(0, N):

        # random index
        ix = np.random.randint(0, N)
        
        # get peptide       
        peptide = peptides[ix]

        # get target prediction value
        y_target = y[ix]
       
        # get initial prediction
        y_pred = np.dot(peptide, weights)

        # gradient descent 
        gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon) # updates weights

    #compute error
    gerr, mse = cumulative_error(peptides, y, lamb, weights) 
    gerror_plot.append(gerr)
    mse_plot.append(mse)
    
    # predict on training data
    train_pred = predict( peptides, weights )
    train_mse = cal_mse( y, train_pred )
    train_mse_plot.append(train_mse)
    train_pcc = pearsonr( y, train_pred )
    train_pcc_plot.append( train_pcc[0] )
        
    # predict on evaluation data
    eval_pred = predict(evaluation_peptides, weights )
    eval_mse = cal_mse(evaluation_targets, eval_pred )
    eval_mse_plot.append(eval_mse)
    eval_pcc = pearsonr(evaluation_targets, eval_pred)
    eval_pcc_plot.append( eval_pcc[0] )
    
    # early stopping
    if early_stopping:
        
        if eval_mse < best_error:
            
            best_error = eval_mse
            best_pcc = eval_pcc[0]
            best_epoch = e
            best_weights[:] = weights[:]
            
            #print ("# Save network", e, "Best MSE", best_error, "PCC", best_pcc)
            
    #print("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)


# ## Error Plot

# In[14]:


# fig = plt.figure(figsize=(10, 10), dpi= 80)

# x = np.arange(0, len(gerror_plot))

# plt.subplot(2, 2, 1)
# plt.plot(x, gerror_plot)
# plt.ylabel("Global Error", fontsize=10);
# plt.xlabel("Iterations", fontsize=10);

# plt.subplot(2, 2, 2)
# plt.plot(x, mse_plot)
# plt.ylabel("MSE", fontsize=10);
# plt.xlabel("Iterations", fontsize=10);


# x = np.arange(0, len(train_mse_plot))

# plt.subplot(2, 2, 3)
# plt.plot(x, train_mse_plot, label="Training Set")
# plt.plot(x, eval_mse_plot, label="Evaluation Set")
# plt.ylabel("Mean Squared Error", fontsize=10);
# plt.xlabel("Iterations", fontsize=10);
# plt.legend(loc='upper right');


# plt.subplot(2, 2, 4)
# plt.plot(x, train_pcc_plot, label="Training Set")
# plt.plot(x, eval_pcc_plot, label="Evaluation Set")
# plt.ylabel("Pearson Correlation", fontsize=10);
# plt.xlabel("Iterations", fontsize=10);
# plt.legend(loc='upper left');


# ## Get PSSM Matrix

# ### Vector to Matrix

# In[15]:


# our matrices are vectors of dictionaries
def vector_to_matrix(vector, alphabet):
    
    rows = int(len(vector)/len(alphabet))
    
    matrix = [0] * rows
    
    offset = 0
    
    for i in range(0, rows):
        
        matrix[i] = {}
        
        for j in range(0, 20):
            
            matrix[i][alphabet[j]] = vector[j+offset] 
        
        offset += len(alphabet)

    return matrix


# ### Matrix to Psi-Blast

# In[16]:


def to_psi_blast(matrix):

    # print to user
    
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for i, row in enumerate(matrix):

        scores = []

        scores.append(str(i+1) + " A")

        for letter in letter_order:

            score = row[letter]

            scores.append(round(score, 4))

        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 


# ### Print

# In[17]:


#matrix = vector_to_matrix(weights, alphabet)
#to_psi_blast(matrix)


# In[18]:


#print(weights)
#print("best")
#print(best_weights)


# ## Performance Evaluation

# In[19]:


#evaluation_peptides = evaluation[:, 0]
#evaluation_peptides = np.array(encode(evaluation_peptides, scheme, alphabet))

# evaluation_targets = np.array(evaluation[:, 1], dtype=float)

# y_pred = []
# for i in range(0, len(evaluation_peptides)):
    # y_pred.append(np.dot(evaluation_peptides[i].T, best_weights))

# y_pred = np.array(y_pred)


# # In[20]:


# pcc = pearsonr(evaluation_targets, np.array(y_pred))
# print("PCC: ", pcc[0])

# plt.scatter(y_pred, evaluation_targets);


# In[21]:


validation_peptides = validation[:, 0]
validation_peptides = np.array(encode(validation_peptides, scheme, alphabet))

validation_targets = np.array(validation[:, 1], dtype=float)

y_pred = []
for i in range(0, len(validation_peptides)):
    y_pred.append(np.dot(validation_peptides[i].T, best_weights))

y_pred = np.array(y_pred)


# In[22]:


pcc = pearsonr(validation_targets, np.array(y_pred))
print("PCC: ", pcc[0])


