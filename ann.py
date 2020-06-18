#!/bin/python3

"""
Script for training neural network

@author: hmmartiny
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(
        "Train an artificial neural network (ANN).\nExample:\n python3 ann.py -t data/A0201/f000 -v data/A0201/f001 --blosum-file data/BLOSUM50 -u 16 -e 100 -n test.net --verbose",
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose'
    )

    # data args
    data_args = parser.add_argument_group("Data files")
    data_args.add_argument(
        '-t', '--train_data',
        type=str,
        required=True,
        help='Data with training data',
        dest='train_data'
    )
    data_args.add_argument(
        '-v', '--valid_data',
        type=str,
        required=True,
        help='Data with validation data',
        dest='valid_data'
    )
    data_args.add_argument(
        '--blosum-file',
        type=str,
        required=True,
        help='Blosum file',
        dest='blosum_file'
    )
    data_args.add_argument(
        '-n', '--net_file',
        type=str,
        required=True,
        help='Name of file to save network to',
        dest='net_file'
    )

    # ANN specific args
    ann_args = parser.add_argument_group('ANN arguments')
    ann_args.add_argument(
        '-u', '--units',
        type=int,
        nargs='+',
        help='Number of units in each layer, e.g. two hidden layers will be -u 16 32',
        required=True,
        dest='out_units'
    )
    ann_args.add_argument(
        '-p', '--p_dropout',
        type=float,
        help='Probability of droput (by default 0)',
        default=0,
        dest='p_dropout'
    )
    ann_args.add_argument(
        '-a', '--activation',
        type=str,
        help='Type of activation function in hidden layers',
        choices=['relu', 'leaky_relu', 'tanh'],
        dest='activation_function',
        default='relu'
    )

    # Training params
    train_args = parser.add_argument_group('Arguments for training the neural network')
    train_args.add_argument(
        '-e', '--epochs',
        type=int,
        default=1000,
        help='Number of epochs to train the network in (by default 1000)',
        dest='epochs'
    )
    train_args.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate for optimizer (by default 0.01)',
        dest='learning_rate'
    )
    train_args.add_argument(
        '-el', '--early_stopping',
        action='store_true',
        help='Use Early Stopping',
        dest='early_stopping'
    )
    train_args.add_argument(
        '-np', '--n_patience',
        type=int,
        default=10,
        dest='n_patience',
        help='patience=epochs/n_patience (by default n_patience=10)'
    )
    train_args.add_argument(
        '-bs', '--batch_size',
        type=int,
        default=16
    )
    train_args.add_argument(
        '-o', '--optimizer',
        type=str,
        default='SGD',
        help='Name of optimizer (by default SGD)',
        choices=['SGD', 'Adam'],
        dest='optimizer'
    )
    train_args.add_argument(
        '-m', '--momentum',
        type=float,
        default=0,
        help='Momentum factor (by default 0)',
        dest='momentum'
    )
    train_args.add_argument(
        '-l2', '--l2_reg',
        type=float,
        default=0,
        help='Weight decay (L2 penalty) for optimizer (by default 0)',
        dest='l2_reg'

    )
    
    return parser.parse_args()

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

def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True

class ANN(nn.Module):
    r"""An artificial neural network (ANN) for predicting 

    Parameters
    ----------
    in_features : int
        Number of features in input
    out_units : list
        List of layers with each element detailing number of neurons in each layer, e.g. two hidden layers: [16, 32]
    n_out : int
        Number of units in output layer
    p_dropout : float
        Probability of dropout, by default 0
    activation : nn.Activation
        A PyTorch activation function, by default nn.ReLU()

    Examples
    ----------
        >>> net = ANN(in_features=189, out_units=[16], n_out=1)
        >>> print(net)
            ANN(
                (fc): Sequential(
                    (fc0): Linear(in_features=189, out_features=16, bias=True)
                    (act0): ReLU()
                    (dropout): Dropout(p=0, inplace=False)
                )
                (fc_out): Linear(in_features=16, out_features=1, bias=True)
            )
    """

    def __init__(self, in_features, out_units, n_out, p_dropout=0, activation=nn.ReLU()):
        super(ANN, self).__init__()

        # save args
        self.in_features = in_features
        self.out_units = out_units
        self.in_units = [self.in_features] + self.out_units[:-1]
        self.n_layers = len(self.out_units)
        self.n_out = n_out

        # build the input and hidden layers
        self.fc = nn.Sequential()
        def add_linear(i):
            """Add n linear layers to the ANN"""
            self.fc.add_module('fc{}'.format(i), nn.Linear(self.in_units[i], self.out_units[i]))
            self.fc.add_module('{}{}'.format('act', i), activation)

        for i in range(self.n_layers):
            add_linear(i)
        
        # add dropout before final
        self.fc.add_module('dropout', nn.Dropout(p_dropout))

        # add final output layer
        self.fc_out = nn.Linear(self.out_units[-1], self.n_out)
    
    def forward(self, x): # (batch_size, in_units)

        o = self.fc(x) # (batch_size, out_units)
        o = self.fc_out(o) # (batch_size, n_out)

        return o

def train(net, train_loader, valid_loader, epochs, optimizer='SGD', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, n_patience=10, use_early_stopping=False, verbose=False):
    """Train a neural network

    Parameters
    ----------
    net : nn.Module
        The neural network
    train_loader : [type]
        iterator for training data
    valid_loader : [type]
        iterator for validation data
    epochs : int
        Number of epochs (iterations) to train the network
    optimizer : optim.Optimizer
        Approach for minimizing loss function
    lr : float
        Learning rate, by default 0.01
    momentum : float
        Momentum factor, by default 0
    l2_reg : float
        Weight decay (L2 penalty) for optimizer
    use_cuda : bool, optional
        Train network on CPU or GPU, by default False
    n_patience : int, optional
        Patience for early stopping: epochs // n_patience, by default 10
    use_early_stopping : bool, optional
        If True, use early stopping. By default False
    verbose : bool, optional
        Print statements if True, by default False

    Returns
    ----------
    net : nn.Module 
        Trained network
    train_loss : list
        Training loss per epoch
    valid_loss : 
        Validation loss per epoch
    """

    # switch to GPU if given
    if use_cuda:
        net.cuda()

    # initializer optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2_reg, momentum=momentum)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
    
    # loss function
    criterion = nn.MSELoss()

    train_loss, valid_loss = [], []

    # early stopping function
    patience = epochs // n_patience
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    if args.verbose and use_early_stopping:
        print("Using early stopping with patience:", patience)

    # start main loop
    for epoch in range(epochs):

        # train
        net.train()

        running_train_loss = 0
        for x_train, y_train in train_loader:
            pred = net(x_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.data
        
        train_loss.append(running_train_loss / len(train_loader))

        # validation
        net.eval()
        
        running_valid_loss = 0
        for x_valid, y_valid in valid_loader:
            pred = net(x_valid)
            loss = criterion(pred, y_valid)
            running_valid_loss += loss.data
        
        valid_loss.append(running_valid_loss / len(valid_loader))

        if verbose:
            print("Epoch {}: Train loss: {:.5f} Valid loss: {:.5f}".format(epoch+1, train_loss[epoch], valid_loss[epoch]))

        if invoke(early_stopping, valid_loss[-1], net, implement=use_early_stopping):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break

    return net, optimizer, train_loss, valid_loss


def data_loader(X_train, X_valid, y_train, y_valid, batch_size=64):
    """Convert numpy arrays into iterable dataloaders with tensors"""

    # convert to tensors
    X_train = torch.from_numpy(X_train.astype('float32'))
    X_valid = torch.from_numpy(X_valid.astype('float32'))
    y_train = torch.from_numpy(y_train.astype('float32')).view(-1, 1)
    y_valid = torch.from_numpy(y_valid.astype('float32')).view(-1, 1)

    # define loaders
    train_loader = DataLoader(
        dataset=TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=TensorDataset(X_valid, y_valid),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, valid_loader

actname2func = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh()
}

if __name__ == "__main__":
    args = parse_args()

    # load data files
    train_raw = load_peptide_target(args.train_data)
    valid_raw = load_peptide_target(args.valid_data)

    # encode with blosum matrix
    x_train, y_train = encode_peptides(train_raw, blosum_file=args.blosum_file, batch_size=args.batch_size, n_features=9)
    x_valid, y_valid = encode_peptides(valid_raw, blosum_file=args.blosum_file, batch_size=args.batch_size, n_features=9)

    # reshape
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)

    # to dataloaders
    train_loader, valid_loader = data_loader(x_train, x_valid, y_train, y_valid, batch_size=args.batch_size)

    # initialize net
    n_features = x_train.shape[1]
    net = ANN(in_features=n_features, out_units=args.out_units, n_out=1, p_dropout=args.p_dropout, activation=actname2func[args.activation_function])
    if args.verbose:
        print(net)
    
    # train
    net, optimizer, train_loss, valid_loss = train(
        net=net,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        momentum=args.momentum,
        l2_reg=args.l2_reg,
        n_patience=args.n_patience,
        use_early_stopping=args.early_stopping,
        verbose=args.verbose
    )

    # save
    to_save = {
        'args': args,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss
    }
    torch.save(to_save, args.net_file)
    if args.verbose:
        print("Saved results to:", args.net_file)