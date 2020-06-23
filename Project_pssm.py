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
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.stats import mode
from argparse import ArgumentParser


parser = ArgumentParser(description="Construction of PSSM matrix with beta optimization")
parser.add_argument("-bs", action="store", dest="bs", type=int, default = 50, help="Beta lower range")
parser.add_argument("-be", action="store", dest="be", type=int, default = 250, help="Beta upper range")
parser.add_argument("-bstep", action="store", dest="bstep", type=int, default = 1, help="Step of incrementing beta")
parser.add_argument("-as", action="store", dest="als", type=int, default = 50, help="Alpha lower range")
parser.add_argument("-ae", action="store", dest="ale", type=int, default = 250, help="Alpha upper range")
parser.add_argument("-astep", action="store", dest="alstep", type=int, default = 1, help="Step of incrementing alpha")
parser.add_argument("-outdir", action="store", dest="model_out", type=str, help="Folder to save matrices")


args = parser.parse_args()

bstart = args.bs
bend = args.be
bstep = args.bstep
astart = args.als
aend = args.ale
astep = args.alstep
out = args.model_out


#only param to optimize
beta_vals = list(range(bstart,bend,bstep))
alpha_vals = list(range(astart,aend,astep))

vals = []
for i in beta_vals:
    for k in alpha_vals:
        vals.append((i, k))

outer_cv_partitions = 5
inner_cv_partitions = 4

data_dir = "../Algo/data/"

model_out_prefix = out + 'PPSM_mat'
#model_out_prefix = 'PSSM_out/PPSM_mat' 

#neff is equal to number of clusters, as sum(w) of peptides is cluster = 1
neff = 100

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
def from_psi_blast(file_name, peptide_length):

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


def train(train_peptides, train_targets, test_peptides, test_targets, beta, alpha, opt):
    
    
    peptide_length = len(train_peptides[0])
    for i in range(0, len(train_peptides)):
        if len(train_peptides[i]) != peptide_length:
            print("Error, peptides differ in length!")
        
        
    #count matrix
    c_matrix = initialize_matrix(peptide_length, alphabet)
    
    for position in range(0, peptide_length):
            
        for peptide in train_peptides:
    
            c_matrix[position][peptide[position]] += 1

    
    
    #observed frequencies matrix (f) 
    f_matrix = initialize_matrix(peptide_length, alphabet)
    
    for position in range(0, peptide_length):
      
        n = 0;
      
        for peptide in train_peptides:
        
            f_matrix[position][peptide[position]] += weights[peptide]
        
            n += weights[peptide]
            
        for letter in alphabet: 
            
            f_matrix[position][letter] = f_matrix[position][letter]/n
    
    
    #pseudo frequencies matrix (g)
    g_matrix = initialize_matrix(peptide_length, alphabet)
    
    for position in range(0, peptide_length):
    
        for letter_1 in alphabet:
            for letter_2 in alphabet:
            
                g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]
    
    
    #combined frequencies matrix (p)
    p_matrix = initialize_matrix(peptide_length, alphabet)

    
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
    if not opt:
        file_name = model_out_prefix + f"_{outer_iteration}.tab"
        to_psi_blast_file(w_matrix, file_name)
    
      
    test_predictions = []
    for test_peptide in test_peptides:
        test_predictions.append(score_peptide(test_peptide, w_matrix))


    pcc_test = spearmanr(test_targets, test_predictions)


    return pcc_test


#main

             
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

 

best_pcc_eval = (-np.inf, -1)

model_track = []
test_indices = []
beta_track = []
alpha_track = []

#load dataset
dataset_file = "final_data.txt"
dataset_all = pd.read_csv(dataset_file, dtype=str)

#assign weights based on hobohm 1 clustering
weights = {}
for i in range(len(dataset_all)):
    weights[dataset_all.iloc[i]['peptide']] = float(dataset_all.iloc[i]['weight'])


for outer_iteration in range(outer_cv_partitions):
    

    evalset = dataset_all[dataset_all['evalset'] == str(outer_iteration)]
    dataset = dataset_all[dataset_all['evalset'] != str(outer_iteration)]
    
    #keep track on used peptides in test set so they are not used again in another test set
    test_indices = []
    
    len_partition = int(len(dataset)/inner_cv_partitions)
    
    
    best_pcc_test = (-np.inf, -1)
    
    beta_opt = []
    alpha_opt = []
    
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
        train_peptides = dataset_train['peptide'].values
        train_targets = dataset_train['target'].astype(float).values
        test_peptides = dataset_test['peptide'].values
        test_targets = dataset_test['target'].astype(float).values
        
        #print(f'Inner {iteration}')
        
        for opt in range(len(vals)):
            
            beta = vals[opt][0]
            alpha = vals[opt][1]
            
            pcc = train(train_peptides, train_targets, test_peptides, test_targets, beta, alpha, opt)
                    
            if pcc[0] > best_pcc_test[0]:
                best_pcc_test = (pcc[0], opt)


        print(f'\nIteration {outer_iteration} {iteration}')
        print("Best PCC test: ", best_pcc_test[0])
        print('Beta =', vals[best_pcc_test[1]][0])
        print('Alpha =', vals[best_pcc_test[1]][1])
        
        beta_opt.append(vals[best_pcc_test[1]][0])
        alpha_opt.append(vals[best_pcc_test[1]][1])
        model_track.append((outer_iteration, iteration, opt))
    
    
    train_outer_peptides = dataset['peptide'].values
    train_outer_targets = dataset['target'].astype(float).values
    
    eval_peptides = evalset['peptide'].values
    eval_targets = evalset['target'].astype(float).values

    beta = mode(beta_opt)[0][0]
    beta_track.append(beta)
    alpha = mode(alpha_opt)[0][0]
    alpha_track.append(alpha)
    
    pcc = train(train_outer_peptides, train_outer_targets, eval_peptides, eval_targets, beta, alpha, False)
    
    matrix_file = model_out_prefix + f"_{outer_iteration}.tab"
    matrix = from_psi_blast(matrix_file, len(train_outer_peptides[0]))
    
    eval_predictions = []
    for eval_peptide in eval_peptides:
        eval_predictions.append(score_peptide(eval_peptide, matrix))

    
    pcc_eval = spearmanr(eval_targets, eval_predictions)
    print(f'\nIteration {outer_iteration}, beta={beta}, alpha={alpha}')
    print("PCC eval: ", pcc_eval[0])

    
    if pcc_eval[0] > best_pcc_eval[0]:
        best_pcc_eval = (pcc_eval[0], outer_iteration)


best_model_ind = best_pcc_eval[1]
print(f'\nBest model: Outer iteration {best_model_ind}, beta {beta_track[best_model_ind]}, alpha {alpha_track[best_model_ind]}')
print(f'Best PCC eval: {best_pcc_eval[0]}')

