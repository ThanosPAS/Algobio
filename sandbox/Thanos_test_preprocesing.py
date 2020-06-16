# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:21:01 2020

@author: white
"""

import numpy as np
import pandas as pd



MAX_PEP_SEQ_LEN = 9

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[5,7], names=['peptide','target'])
    return df[df.peptide.apply(len) <= MAX_PEP_SEQ_LEN]


train_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algobio/data/predictions_smm.txt"

train_raw = load_peptide_target(train_data)