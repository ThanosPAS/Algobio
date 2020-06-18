import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser("Train an artificial neural network (ANN)")

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

    # ANN specific args
    ann_args = parser.add_argument_group('ANN arguments')
    ann_args.add_argument(
        '-n_layers',
        default=3
    )
    
    return parser.parse_args()

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

def train(net, train_loader, valid_loader, epochs, optimizer='SGD', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, verbose=False):
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
    verbose : bool, otpional
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
        optimizer = optim.SGD(lr=lr, weight_decay=l2_reg, momentum=momentum)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(lr=lr, weight_decay=l2_reg)
    
    # loss function
    criterion = nn.MSELoss()

    train_loss, valid_loss = [], []

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
    
    return net, train_loss, valid_loss

def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename, MAX_PEP_SEQ_LEN=9):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
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

if __name__ == "__main__":
    #args = parse_args()
    net = ANN(9, [16, 32], 1)
    print(net)