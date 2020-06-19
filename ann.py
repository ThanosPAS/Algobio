#!/bin/python3

"""
Script for training neural network

@author: hmmartiny
"""

import argparse
import itertools
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from pytorchtools import EarlyStopping

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

def invoke(early_stopping, loss, model, implement=False, verbose=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            if verbose:
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

def _train(net, train_loader, Eval_loader, epochs, optimizer='SGD', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, n_patience=10, use_early_stopping=False, verbose=False):
    """Train a neural network

    Parameters
    ----------
    net : nn.Module
        The neural network
    train_loader : [type]
        iterator for training data
    Eval_loader : [type]
        iterator for Evalation data
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
    Eval_loss : 
        Evalation loss per epoch
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

    train_loss, Eval_loss = [], []

    # early stopping function
    patience = epochs // n_patience
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    if verbose and use_early_stopping:
        print("Using early stopping with patience:", patience)

    # start main loop
    for epoch in range(epochs):

        # train
        net.train()

        running_train_loss = 0
        for X, y in train_loader:
            pred = net(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.data
        
        train_loss.append(running_train_loss / len(train_loader))

        # Evalation
        net.eval()
        
        running_Eval_loss = 0
        for x_Eval, y_Eval in Eval_loader:
            pred = net(x_Eval)
            loss = criterion(pred, y_Eval)
            running_Eval_loss += loss.data
        
        Eval_loss.append(running_Eval_loss / len(Eval_loader))

        if verbose:
            print("Epoch {}: Train loss: {:.5f} Eval loss: {:.5f}".format(epoch+1, train_loss[epoch], Eval_loss[epoch]))

        if invoke(early_stopping, Eval_loss[-1], net, implement=use_early_stopping):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break

    return net, optimizer, train_loss, Eval_loss, epoch


def data_loader(X, X_Eval, y, y_Eval, batch_size=64):
    """Convert numpy arrays into iterable dataloaders with tensors"""

    # convert to tensors
    X = torch.from_numpy(X.astype('float32'))
    X_Eval = torch.from_numpy(X_Eval.astype('float32'))
    y = torch.from_numpy(y.astype('float32')).view(-1, 1)
    y_Eval = torch.from_numpy(y_Eval.astype('float32')).view(-1, 1)

    # define loaders
    train_loader = DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True
    )
    Eval_loader = DataLoader(
        dataset=TensorDataset(X_Eval, y_Eval),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, Eval_loader

actname2func = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh()
}

def train_cv(X, X_test, y, y_test, batch_size=16, epochs=100, out_units=[16], optimizer='SGD', p_dropout=0, activation_function='relu', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, n_patience=10, verbose=False, use_early_stopping=False):
    """Wrapper function to use in nested CV"""

    if not isinstance(out_units, list):
        out_units = [out_units]

    # convert to PyTorch datasets 
    train_loader, Eval_loader = data_loader(X, X_test, y, y_test, batch_size=batch_size)

    # define neural network
    in_features = X.shape[1]
    net = ANN(in_features=in_features, out_units=out_units, n_out=1, p_dropout=p_dropout, activation=actname2func[activation_function])
    if verbose:
        print(net)


    net, optimizer, train_loss, Eval_loss, epoch = _train(
        net=net,
        train_loader=train_loader,
        Eval_loader=Eval_loader,
        epochs=epochs,
        optimizer=optimizer,
        lr=lr,
        momentum=momentum,
        l2_reg=l2_reg,
        n_patience=n_patience,
        use_early_stopping=use_early_stopping,
        verbose=verbose,
        use_cuda=torch.cuda.is_available()
    )

    final_train_loss = train_loss[epoch] # [-1] also works instead of indexing by epoch
    final_eval_loss = Eval_loss[epoch]

    return final_train_loss, final_eval_loss, net

if __name__ == "__main__":

    # load data
    train_raw = load_peptide_target('data/A0201/A0201.dat')

    # encode with blosum matrix 
    X, y = encode_peptides(train_raw, blosum_file='data/BLOSUM50', batch_size=32, n_features=9)
    #x_Eval, y_Eval = encode_peptides(Eval_raw, blosum_file='data/BLOSUM50', batch_size=32, n_features=9)

    # reshape
    X = X.reshape(X.shape[0], -1)
    #x_Eval = x_Eval.reshape(x_Eval.shape[0], -1)    

    # param grids
    param_grids = {
        'out_units': [4], #np.arange(4, 21, step=4),
        'lr': [0.1],#, 0.01, 0.001],
        'optimizer': ['SGD'], # , 'Adam'],
        'p_dropout': [0], #np.arange(0, .5, step=0.1),
        'epochs': [1000],
        'use_early_stopping': [True]#, False]
    }
    # get all combinations of hyperparameters given in param_grids
    k, v = zip(*param_grids.items())
    params = [dict(zip(k,v)) for v in itertools.product(*v)]

    # define inner and outer partitions
    K1, K2 = 2, 2
    CV1 = model_selection.KFold(n_splits=K1,shuffle=True, random_state = 42)
    CV2= model_selection.KFold(n_splits=K2,shuffle=True, random_state = 42)
    
    outer_val_losses = []

    # Outer CV
    for k, (train_index, test_index) in enumerate(CV1.split(X, y)):
    
        # extract training and test set for current CV fold
        Xr_train = X[train_index,:]
        yr_train = y[train_index]
        Xr_test = X[test_index,:]
        yr_test = y[test_index]
        
        inner_test_loss, outer_Eval_loss = [], []
        
        print('Outer CV fold {0}/{1}'.format(k+1,K1))
        
        # Inner CV
        inner_val_losses = []
        for j, (train_index, test_index) in enumerate(CV2.split(Xr_train)):
            print("> Inner CV fold {}/{}".format(j+1, K2))
        
            # extract training and test set for current CV fold
            Xin_train, yin_train = Xr_train[train_index,:], yr_train[train_index]
            Xin_test, yin_test = Xr_train[test_index,:], yr_train[test_index]

            # 
            min_val_loss = np.inf
            best_params = {}

            # test set of params from grid
            for param_set in params:
                print(">> params:", param_set)
                inner_train_loss, inner_eval_loss, _ = train_cv(
                    X=Xin_train, 
                    X_test=Xin_test, 
                    y=yin_train, 
                    y_test=yin_test,
                    **param_set
                )
                print(">>> Train Loss: {:.3f}, Eval Loss: {:.3f}".format(inner_train_loss, inner_eval_loss)) 
                # see if the error is better now
                if inner_eval_loss < min_val_loss:
                    min_val_loss = inner_eval_loss
                    best_params = param_set
            
            inner_val_losses.append((min_val_loss, best_params))

        # select best set of params from inner folds
        best_inner_loss, best_inner_params = min(inner_val_losses, key=itemgetter(0))
        
        print("Best inner eval loss: {:.5f} with params {}".format(best_inner_loss, best_inner_params))

        # Train with optimal hyperparameter search
        outer_train_loss, outer_eval_loss, net = train_cv(
            X=Xr_train, 
            X_test=Xr_test, 
            y=yr_train, 
            y_test=yr_test,
            **best_inner_params
            )
        
        # Compute test error on outer partition
        outer_val_losses.append((outer_train_loss, outer_eval_loss, best_inner_params, net))
        print("Score on outer test: Train loss: {:.3f}, Eval loss: {:.3f}".format(outer_train_loss, outer_eval_loss)) 
    
    
    # Compute average errors for the outer folds
    o_train_loss, o_valid_loss, o_best_params, _ = zip(*outer_val_losses)
    print("\nTrain loss {:.3f} (+/- {:.3f}), Eval loss {:.3f} (+/- {:.3f})".format(
        np.mean(o_train_loss), np.std(o_train_loss),
        np.mean(o_valid_loss), np.std(o_valid_loss)
    ))

    # What was the set of parameters that gave the lowest error?
    best_results = min(outer_val_losses, key=itemgetter(1))
    print("Best params:", best_results[2])

    # Print out results of the outer folds nicely
    for i, outer_results in enumerate(outer_val_losses):
        print("Outer CV {}: Train loss: {:.3f}, Eval loss: {:.3f}, params: {}".format(
            i+1, o_train_loss[i], o_valid_loss[i], o_best_params[i]
        ))

    # save best net
    best_net = best_results[3]
    torch.save(best_net, 'data/best_ann.net')
