#!/bin/python3

import numpy as np
import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("Concat data")

    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        help='Path to data dir with files to be concatenated',
        required=True,
        dest='data_dir'
    )

    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        dest='output_file',
        help='Output file concatenated data is stored in'
    )
    
    return parser.parse_args()

def load_peptide_target(filename, MAX_PEP_SEQ_LEN=9):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    # load
    df = pd.read_csv(filename, sep=' ', usecols=[0,1], names=['peptide','target'])

    # filter by peptide length
    df = df.loc[df.peptide.str.len() <= MAX_PEP_SEQ_LEN]
    return df

if __name__ == "__main__":
    args = parse_args()

    files = [x for x in os.listdir(args.data_dir) if not x.endswith('.dat')]

    # loop through files in datadir and load data
    outs = [] 
    for f in files:
        f = os.path.join(args.data_dir, f)
        d = load_peptide_target(f)
        outs.append(d)
    
    # concat to one df
    df = pd.concat(outs, axis=0)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # write to output file
    df.to_csv(args.output_file, index=None)

    print("CSV file saved:", args.output_file)
    print("Shape:", df.shape)
    print(df.head())

# python preprocessing.py -d data/A0201/ -o data/A0201.csv


    