

import numpy as np
import pandas as pd



MAX_PEP_SEQ_LEN = 9

files = ["c001","c002","c003","c004","f000","f001","f002","f003","f004"]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
    return df[df.peptide.apply(len) <= MAX_PEP_SEQ_LEN]



train_raw = pd.read_csv('c000', sep='\s+', usecols=[0,1], names=['peptide','target'])
for batch in files:
    #train_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/A0201/"
    train_data = batch
    pre_train = load_peptide_target(train_data)
    train_raw = pd.concat([train_raw,pre_train], ignore_index = True)
    pre_train.iloc[0:0]

train_raw.sort_values(by=['target'], inplace=True, ascending=False)
train_raw.drop_duplicates(inplace = True)
train_raw.index = range(train_raw.shape[0])



#blosum_file = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/BLOSUM50"
#train_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/A0201/f000"
valid_data = pd.read_csv('f001', sep='\s+', usecols=[0,1], names=['peptide','target'])
test_data = pd.read_csv('c000', sep='\s+', usecols=[0,1], names=['peptide','target'])



valid_data.drop_duplicates(inplace = True)
test_data.drop_duplicates(inplace = True)