

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



train_raw = pd.read_csv("C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/A0201/c000", sep='\s+', usecols=[0,1], names=['peptide','target'])
#train_raw = train_raw.fillna(0)
for batch in files:
    train_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/A0201/"
    train_data = train_data + batch
    pre_train = load_peptide_target(train_data)
    train_raw = pd.concat([train_raw,pre_train], ignore_index = True)
    pre_train.iloc[0:0]

train_raw.sort_values(by=['target'], inplace=True, ascending=False)
train_raw.drop_duplicates(inplace = True)
train_raw.index = range(train_raw.shape[0])

#duplicated = train_raw[train_raw.duplicated()]
    