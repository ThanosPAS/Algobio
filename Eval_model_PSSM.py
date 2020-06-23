#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:10:45 2020

@author: kostas
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

sns.set(style='whitegrid')

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


# BINDER_THRESHOLD (from Helle - I guess we use this, but not sure what the best argumentation would be)
BINDER_THRESHOLD = 0.426

df = pd.read_csv('final_data.txt')
#eval_dataset = df[df['evalset'] == 2]
eval_dataset = df
eval_peptides = eval_dataset['peptide'].values
eval_targets = eval_dataset['target'].astype(float).values

matrix = from_psi_blast('./Final_models/PPSM_mat_2.tab', 9)

eval_predictions = []
for eval_peptide in eval_peptides:
    eval_predictions.append(score_peptide(eval_peptide, matrix))

eval_predictions = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(eval_predictions).reshape(-1, 1)).reshape(1,-1)[0]
eval_predictions = np.array(eval_predictions)

plt.scatter(eval_targets, eval_predictions)
plt.xlabel('Targets')
plt.xlabel('Predictions')

plt.savefig('Scatter_PSSM.png', format = 'png', dpi = 300)
plt.show()

y_test_class = np.where(eval_targets>= BINDER_THRESHOLD, 1, 0)
y_pred_class = np.where(eval_predictions >= BINDER_THRESHOLD, 1, 0)

fpr, tpr, threshold = roc_curve(y_test_class, y_pred_class)
roc_auc = auc(fpr, tpr)

def plot_roc_curve():
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('AUC_PSSM.png', format = 'png', dpi = 300)
    plt.show()

plot_roc_curve()

scc = stats.spearmanr(eval_targets, eval_predictions)
def plot_scc():
    plt.title('Spearmans Correlation Coefficient')
    plt.scatter(eval_targets, eval_predictions, label = 'SCC = %0.2f' % scc[0])
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.savefig('SCC_PSSM.png', format = 'png', dpi = 300)
    plt.show()

plot_scc()

mcc = matthews_corrcoef(y_test_class, y_pred_class)
def plot_mcc():
    plt.title('Matthews Correlation Coefficient')
    plt.scatter(eval_targets, eval_predictions, label = 'MCC = %0.2f' % mcc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted') 
    plt.xlabel('Validation targets')
    plt.savefig('MCC_PSSM.png', format = 'png', dpi = 300)
    plt.show()

plot_mcc()
    
# =============================================================================
# #fill out
# x_test =
# y_test =
# pred =
# targets =
# 
# ### ROC/AUC
# ## net.eval depending on method - here ANN
# net.eval()
# pred = net(x_test)
# loss = criterion(pred, y_test)
# 
# plot_target_values(data=[(pd.DataFrame(pred.data.numpy(), columns=['target']), 'Prediction'),
#                          (test_raw, 'Target')])
# 
# y_test_class = np.where(y_test.flatten() >= BINDER_THRESHOLD, 1, 0)
# y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)
# 
# fpr, tpr, threshold = roc_curve(y_test_class, pred.flatten().detach().numpy())
# roc_auc = auc(fpr, tpr)
# 
# 
# def plot_roc_curve():
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
# 
# plot_roc_curve()
# 
# 
# ### SCC and PCC
# 
# scc = stats.spearman(pred, targets)
# def plot_scc():
#     plt.title('Spearmans Correlation Coefficient')
#     plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'SCC = %0.2f' % scc)
#     plt.legend(loc = 'lower right')
#     plt.ylabel('Predicted')
#     plt.xlabel('Validation targets')
#     plt.show()
# 
# 
# pcc = stats.pearsonr(pred, targets)
# def plot_scc():
#     plt.title('Pearsons Correlation Coefficient')
#     plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'PCC = %0.2f' % pcc)
#     plt.legend(loc = 'lower right')
#     plt.ylabel('Predicted')
#     plt.xlabel('Validation targets')
#     plt.show()
# 
# 
# ### Spearman
# if stats.shapiro(pred)[1] < 0.05 and stats.shapiro(targets)[1] < 0.05:
#     print('Preds and targets normally distributed (according to shapiro)')
#     print('PCC', stats.pearsonr(pred, targets))
#     lot_pcc()
#     print('SCC', stats.spearman(pred, targets))
#     plot_scc()
# else:
#     print('Preds and targets not normally distributed (according to shapiro) -> only do Spearman')
#     print('SCC', stats.spearman(pred, targets))
#     plot_scc()
# 
# 
# =============================================================================
