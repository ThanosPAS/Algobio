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
import pickle
import pandas as pd
from itertools import combinations


# ## Data Imports

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[2]:


data_dir = "data/"


# ### Hannah's partition scheme and loading

# In[3]:


def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=' ', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename, MAX_PEP_SEQ_LEN=9):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep=' ', usecols=[0,1], names=['peptide','target'])
    return df[df.peptide.apply(len) <= MAX_PEP_SEQ_LEN]

def load_pickle(f):
    with open(f, 'rb') as source:
        s = pickle.load(source)
    return s

def load_partitions(files):
    o = []
    for f in files:
        data = load_pickle(f)
        o += data
    return o

def assign_cv_partition(partition_files, n_folds=5, n_test=1):
    """Figure out all combinations of partition_files to assign as train and test data in CV"""

    # how many combinations of partition_files in train part
    n_train = n_folds - n_test

    # find all combinations of the partition_files with n_train files in each
    train_files = list(combinations(partition_files, n_train))

    # convert each list element to tuple so (train_partitions, test_partition)
    files = [
        (x, list(set(partition_files) - set(x))) for x in train_files
    ]

    return files

def data_partition(partition_files, data, blosum_file, batch_size=32, n_features=9):
    partitions = load_partitions(partition_files)

    selected_data = data.loc[data.peptide.isin(partitions), ].reset_index()

    X, y = encode_peptides(selected_data, blosum_file=blosum_file, batch_size=batch_size, n_features=n_features)

    # reshape X
    X = X.reshape(X.shape[0], -1)

    return X, y

def encode_peptides(Xin, blosum_file, batch_size, n_features, MAX_PEP_SEQ_LEN=9):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (batch_size, MAX_PEP_SEQ_LEN, n_features)
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(Xin)
    n_features = len(blosum)
    
    Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=np.int8) # should it be uint? is there a purpose to that?
    
    for peptide_index, row in Xin.iterrows():
        for aa_index in range(len(row.peptide)):
            Xout[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
            
    return Xout, Xin.target.values

data = load_peptide_target('data/A0201/A0201.dat')


# ### Alphabet

# In[4]:


alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)


# ## Error Function

# In[5]:


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

# In[6]:


def predict(peptides, weights):

    pred = []
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]
        
        # get prediction
        y_pred = np.dot(peptide/5, weights)
        
        pred.append(y_pred)
        
    return pred


# ## Calculate MSE between two vectors

# In[7]:


def cal_mse(vec1, vec2):
    
    mse = 0
    
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
        
    mse /= len(vec1)
    
    return( mse)


# ## Gradient Descent

# In[8]:


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

# In[9]:


#Lambdas
lamb_list=[10,1,0.1,0.01,0.001]

# learning rate, epsilon
epsilon_list = [0.0005, 0.00005, 0.000005]

# training epochs
epochs = 200

#Use Hannahs encoding loading
encoding_file = 'data/BLOSUM50' # could change it to data/sparse

# Random seed 
np.random.seed( 1 )

# early stopping
early_stopping = True

# partition files
partition_files = ['data/partition_3.txt', 'data/partition_2.txt', 'data/partition_6.txt', 'data/partition_5.txt', 'data/partition_4.txt']

K1, K2 = 5, 4
    
# define outer partitions
outer_partitions = assign_cv_partition(partition_files, n_folds=K1)
for k, (outer_train, outer_test) in enumerate(outer_partitions):
    
    #get outer training set from outer partition for training with optimal parameters
    outer_peptides, outer_y = data_partition(outer_train, data=data, blosum_file=encoding_file)
    
    #get validation set from the outer partition to validate model one
    validation_peptides, validation_targets = data_partition(outer_test, data=data, blosum_file=encoding_file)
    
    # make inner partition of the training set for parameter optimsiation
    inner_partitions = assign_cv_partition(outer_train, n_folds=K2)
    
    #initial best performance pcc
    best_pcc=0
    for j, (inner_train, inner_test) in enumerate(inner_partitions):

        # peptides for training
        peptides, y = data_partition(outer_train, data=data, blosum_file=encoding_file)
        N = len(peptides)
        
        
        # target values
        evaluation_peptides, evaluation_targets = data_partition(outer_test, data=data, blosum_file=encoding_file)


        #for each lambda
        for l in lamb_list:
            for epsilon in epsilon_list:
                lamb_N = l/N
                stopping_error = np.inf # error for early stopping
                # weights
                input_dim  = len(peptides[0])
                output_dim = 1
                w_bound = 0.1
                weights = np.random.uniform(-w_bound, w_bound, size=input_dim)                


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
                        #print(y_target)
                        # get initial prediction
                        y_pred = np.dot(peptide, weights)
                        #print(y_pred)
                        # gradient descent 
                        gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon) # updates weights

                    #compute error
                    gerr, mse = cumulative_error(peptides, y, l, weights) 

                    # predict on training data
                    train_pred = predict( peptides, weights )
                    train_mse = cal_mse( y, train_pred )
                    train_pcc = pearsonr( y, train_pred )

                    # predict on evaluation data
                    eval_pred = predict(evaluation_peptides, weights )
                    eval_mse = cal_mse(evaluation_targets, eval_pred )
                    eval_pcc = pearsonr(evaluation_targets, eval_pred)

                    # early stopping
                    if early_stopping:

                        if eval_mse < stopping_error:

                            stopping_error = eval_mse # save to compare future loops
                            stopping_pcc = eval_pcc[0] # save to compare with best pcc
                            

                            #print ("# Save network", e, "Best MSE", stopping_error, "PCC", stopping_pcc)
                    if stopping_pcc > best_pcc:
                        best_pcc = stopping_pcc
                        best_lamb = l
                        best_epsilon = epsilon
                    
                    #print("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)
                print("Lambda ", l, "Epsilon ",epsilon,"PCC ",stopping_pcc, "Outer ",k, "Inner",j)
    # train on outer test set
    lamb=best_lamb
    lamb_N = lamb/N
    epsilon=best_epsilon
    stopping_error = np.inf # for early stopping
    # weights
    input_dim  = len(outer_peptides[0])
    output_dim = 1
    w_bound = 0.1
    weights = np.random.uniform(-w_bound, w_bound, size=input_dim)
    best_weights = np.zeros(input_dim)
                


    # for each training epoch
    for e in range(0, epochs):

        # for each peptide
        for i in range(0, N):

            # random index
            ix = np.random.randint(0, N)

            # get peptide       
            peptide = outer_peptides[ix]

            # get target prediction value
            y_target = outer_y[ix]
            #print(y_target)
            # get initial prediction
            y_pred = np.dot(peptide, weights)
            #print(y_pred)
            # gradient descent 
            gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon) # updates weights

        #compute error
        gerr, mse = cumulative_error(outer_peptides, outer_y, lamb, weights) 

        # predict on training data
        train_pred = predict( outer_peptides, weights )
        train_mse = cal_mse( outer_y, train_pred )
        train_pcc = pearsonr( outer_y, train_pred )

        # predict on outer test (validation data)
        eval_pred = predict(validation_peptides, weights )
        eval_mse = cal_mse(validation_targets, eval_pred )
        eval_pcc = pearsonr(validation_targets, eval_pred)

                    # early stopping
        if early_stopping:

            if eval_mse < stopping_error:

                stopping_error = eval_mse # save to compare future loops
                stopping_pcc = eval_pcc[0]
                best_weights[:] = weights[:]
            
    y_pred = []
    for i in range(0, len(validation_peptides)):
        y_pred.append(np.dot(validation_peptides[i].T, best_weights))

    y_pred = np.array(y_pred)

    pcc = pearsonr(evaluation_targets, np.array(y_pred))
    print("Lambda: ", lamb,"Epsilon: ", epsilon, "PCC: ", pcc[0])
    print(best_weights)

