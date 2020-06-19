#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:51:57 2020

@author: kostas
"""


import os
import math
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


#evaluation
def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum


#initialize Matrix
def initialize_matrix(peptide_length, alphabet):

    init_matrix = [0]*peptide_length

    for i in range(0, peptide_length):

        row = {}

        for letter in alphabet: 
            row[letter] = 0.0

        #fancy way:  row = dict( zip( alphabet, [0.0]*len(alphabet) ) )

        init_matrix[i] = row
        
    return init_matrix
    

def to_psi_blast_file(matrix, file_name):
    
    with open(file_name, 'w') as file:

        header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        file.write ('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n'.format(*header)) 

        letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        for i, row in enumerate(matrix):

            scores = []

            scores.append(str(i+1) + " A")

            for letter in letter_order:

                score = row[letter]

                scores.append(round(score, 4))

            file.write('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n'.format(*scores)) 


#load matrix from PSI-BLAST format
def from_psi_blast(file_name):

    f = open(file_name, "r")
    
    nline = 0
    for line in f:
    
        sline = str.split( line )
        
        if nline == 0:
        # recover alphabet
            alphabet = [str]*len(sline)
            for i in range(0, len(sline)):
                alphabet[i] = sline[i]
                
            matrix = initialize_matrix(peptide_length, alphabet)
        
        else:
            i = int(sline[0])
            
            for j in range(2,len(sline)):
                matrix[i-1][alphabet[j-2]] = float(sline[j])
                
        nline+= 1
            
    return matrix


#main
    
data_dir = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/"
             
#load alphabet
alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

#load background frequencies
bg_file = data_dir + "Matrices/bg.freq.fmt"
_bg = np.loadtxt(bg_file, dtype=float)

bg = {}
for i in range(0, len(alphabet)):
    bg[alphabet[i]] = _bg[i]

#load BLOSUM matrix    
blosum62_file = data_dir + "Matrices/blosum62.freq_rownorm"
_blosum62 = np.loadtxt(blosum62_file, dtype=float).T

blosum62 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum62[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum62[letter_1][letter_2] = _blosum62[i, j]

    
outer_cv_partitions = 5
model_out_prefix = 'PPSM_mat'  
best_mse_eval = (np.inf, -1)
best_pcc_eval = (-np.inf, -1)
#only param to optimize
beta_vals = [50, 100, 150, 200]
inner_model_track = []
    
for outer_iteration in range(outer_cv_partitions):
    
    #load dataset
    dataset_file = f"C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/A0201/f00{outer_iteration}"
    dataset = pd.read_csv(dataset_file, dtype=str, header=None, sep='\s+')
    
    #keep track on used peptides in test set so they are not used again in another test set
    test_indices = []
    
    inner_cv_partitions = 4
    len_partition = int(len(dataset)/inner_cv_partitions)
    
    best_mse_test = (np.inf, -1)
    best_pcc_test = (-np.inf, -1)

    for iteration in range(inner_cv_partitions):
    
        #check which indices are remaining to pick from for testing
        remaining_indices = list(set(list(dataset.index)).difference(set(test_indices)))
        
        #randomly select from indices not used for testing in previous iteration
        test_partition_index = random.sample(remaining_indices, len_partition)

        #add used test partition indices to list for exlusion in next iteration
        for ind in test_partition_index:
            test_indices.append(ind)
        
        #select training and test sets
        dataset_train = dataset.drop(test_partition_index, axis=0)
        dataset_test = dataset.loc[sorted(test_partition_index)]
        
        #get values from dataframes, convert to nparrays 
        train_peptides = dataset_train[0].values
        train_targets = dataset_train[1].astype(float).values
        test_peptides = dataset_test[0].values
        test_targets = dataset_test[1].astype(float).values


        peptide_length = len(train_peptides[0])
        for i in range(0, len(train_peptides)):
            if len(train_peptides[i]) != peptide_length:
                print("Error, peptides differ in length!")
        
        
        #count matrix
        c_matrix = initialize_matrix(peptide_length, alphabet)
        
        for position in range(0, peptide_length):
                
            for peptide in train_peptides:
        
                c_matrix[position][peptide[position]] += 1
        
        
        #check, sum for each position should be len(peptides)
        #print(sum(c_matrix[0].values()))
        
        
        #sequence weighting
        weights = {}
        
        for peptide in train_peptides:
        
            w = 0.0
            neff = 0.0
            
            for position in range(0, peptide_length):
        
                r = 0
        
                for letter in alphabet:        
        
                    if c_matrix[position][letter] != 0:
                        
                        r += 1
        
                s = c_matrix[position][peptide[position]]
        
                w += 1.0/(r * s)
        
                neff += r
                    
            neff = neff / peptide_length
           
            weights[peptide] = w
            
        
        #check, sum of weights should be len(peptides[0])
        #print(sum(weights.values()))
        
        
        #observed frequencies matrix (f)
        
        f_matrix = initialize_matrix(peptide_length, alphabet)
        
        for position in range(0, peptide_length):
          
            n = 0;
          
            for peptide in train_peptides:
            
                f_matrix[position][peptide[position]] += weights[peptide]
            
                n += weights[peptide]
                
            for letter in alphabet: 
                
                f_matrix[position][letter] = f_matrix[position][letter]/n
        
        
        #check, sum of pos values should be 1
        #print(sum(f_matrix[0].values()))
                  
        
        #pseudo frequencies matrix (g)
        g_matrix = initialize_matrix(peptide_length, alphabet)
        
        for position in range(0, peptide_length):
        
            for letter_1 in alphabet:
                for letter_2 in alphabet:
                
                    g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]
        
        
        #combined frequencies matrix (p)
        p_matrix = initialize_matrix(peptide_length, alphabet)
        
        alpha = neff - 1
        beta = beta_vals[iteration]
        
        for position in range(0, peptide_length):
        
            for a in alphabet:
                p_matrix[position][a] = ((f_matrix[position][a] * alpha) + (g_matrix[position][a] * beta)) / (alpha + beta)
        
        
        #log odds weight matrix (w)
        w_matrix = initialize_matrix(peptide_length, alphabet)
        
        for position in range(0, peptide_length):
            
            for letter in alphabet:
                if p_matrix[position][letter] > 0:
                    w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter]/bg[letter])/math.log(2)
                else:
                    w_matrix[position][letter] = -999.9
        
        
        # Write out PSSM in Psi-Blast format to file
        file_name = model_out_prefix + f"_{outer_iteration}_{iteration}.tab"
        to_psi_blast_file(w_matrix, file_name)
        
        
        train_predictions = []
        for train_peptide in train_peptides:
            train_predictions.append(score_peptide(train_peptide, w_matrix))
          
        test_predictions = []
        for test_peptide in test_peptides:
            test_predictions.append(score_peptide(test_peptide, w_matrix))
        
        
        scaled_train = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(train_predictions).reshape(-1, 1)).reshape(1,-1)[0]
        scaled_test = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(test_predictions).reshape(-1, 1)).reshape(1,-1)[0]
        
        pcc_train = pearsonr(train_targets, scaled_train)
        pcc_test = pearsonr(test_targets, scaled_test)
        mse_train = 1/len(train_targets) * sum( (train_targets - scaled_train)**2 )
        mse_test = 1/len(test_targets) * sum( (test_targets - scaled_test)**2 )
        
        if pcc_test[0] > best_pcc_test[0]:
            best_pcc_test = (pcc_test[0], iteration)
        if mse_test < best_mse_test[0]:
            best_mse_test = (mse_test, iteration)        
    
        print(f'\nIteration {iteration}, beta={beta}')
        print("PCC train: ", pcc_train[0])
        print("MSE train:", mse_train)
        print("PCC test: ", pcc_test[0])
        print("MSE test:", mse_test)
        #plt.scatter(test_targets, scaled_test)
    
    
    best_model_ind = best_pcc_test[1]
    beta = beta_vals[best_model_ind]
    inner_model_track.append(best_model_ind)
    
    matrix = from_psi_blast(model_out_prefix + f"_{outer_iteration}_{best_model_ind}.tab")
    
    print(f'\nSelected model iteration {best_model_ind}')
    
    eval_file = f"C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/A0201/c00{outer_iteration}"
    eval_dataset = pd.read_csv(eval_file, dtype=str, header=None, sep='\s+')
    
    eval_peptides = eval_dataset[0].values
    eval_targets = eval_dataset[1].astype(float).values
    
    eval_predictions = []
    for eval_peptide in eval_peptides:
        eval_predictions.append(score_peptide(eval_peptide, matrix))

    scaled_eval = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(eval_predictions).reshape(-1, 1)).reshape(1,-1)[0]
    
    pcc_eval = pearsonr(eval_targets, scaled_eval)
    mse_eval = 1/len(eval_targets) * sum( (eval_targets - scaled_eval)**2 )
    print(f'\nIteration {outer_iteration}, beta={beta}')
    print("PCC eval: ", pcc_eval[0])
    print("MSE eval:", mse_eval)
    
    if pcc_eval[0] > best_pcc_eval[0]:
        best_pcc_eval = (pcc_eval[0], iteration)
    if mse_eval < best_mse_eval[0]:
        best_mse_eval = (mse_eval, iteration)


best_model_ind = best_pcc_eval[1]
print(f'\nBest model: Outer iteration {best_model_ind}, inner iteration {inner_model_track[best_model_ind]}, beta {beta_vals[inner_model_track[best_model_ind]]}')
