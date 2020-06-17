#!/usr/bin/env python
# coding: utf-8

# # Train a neural network to predict MHC ligands
# The notebook consists of the following sections:
# 
# 0. Module imports, define functions, set constants
# 1. Load Data
# 2. Build Model
# 3. Select Hyper-paramerters
# 4. Compile Model
# 5. Train Model
# 6. Evaluation
# 
# ## Exercise
# 
# The exercise is to optimize the model given in this notebook by selecting hyper-parameters that improve performance. First run the notebook as is and take notes of the performance (AUC, MCC). Then start a manual hyper-parameter search by following the instructions below. If your first run results in poor fitting (the model doesn't learn anything during training) do not dispair! Hopefully you will see a rapid improvement when you start testing other hyper-parameters.
# 
# ### Optimizer, learning rate, and mini-batches
# The [optimizers](https://pytorch.org/docs/stable/optim.html) are different approaches of minimizing a loss function based on gradients. The learning rate determine to which degree we correct the weights. The smaller the learning rate, the smaller corrections we make. This may prolong the training time. To mitigate this, one can train with mini-batches. Instead of feeding your network all of the data before you make updates you can partition the training data into mini-batches and update weigths more frequently. Thus, your model might converge faster. Also small batch sizes use less memory, which means you can train a model with more parameters.
# 
# If you experienced trouble in even training then you might benefit from lowering the learning rate to 0.01 or 0.001 or perhaps even smaller.
# 
# __Optimizers:__
# 1. SGD (+ Momentum)
# 2. Adam
# 3. Try others if you like...
# 
# __Mini-batch size:__
# When you have implemented and tested a smaller learning rate try also implementing a mini-batch of size 512 or 128. In order to set the mini-batch size use the variable MINI_BATCH_SIZE and run train_with_minibatches() instead of train().
# 
# ### Number of hidden units
# Try increasing the number of model parameters (weights), eg. 64, 128, or 512.
# 
# ### Hidden layers
# Add another layer to the network. To do so you must edit the methods of Net()-class.
# 
# ### Parameter initialization
# Parameter initialization can be extremely important.
# PyTorch has a lot of different [initializers](http://pytorch.org/docs/master/nn.html#torch-nn-init) and the most often used initializers are listed below. Try implementing one of them.
# 1. Kaming He
# 2. Xavier Glorot
# 3. Uniform or Normal with small scale (0.1 - 0.01)
# 
# Bias is nearly always initialized to zero using the [torch.nn.init.constant(tensor, val)](http://pytorch.org/docs/master/nn.html#torch.nn.init.constant)
# 
# To implement an initialization method you must uncomment #net.apply(init_weights) and to select your favorite method you must modify the init_weights function.
# 
# ### Nonlinearity
# Non-linearity is what makes neural networks universal predictors. Not everything in our universe is related by linearity and therefore we must implement non-linear activations to cope with that. [The most commonly used nonliearities](http://pytorch.org/docs/master/nn.html#non-linear-activations) are listed below. 
# 1. ReLU
# 2. Leaky ReLU
# 3. Sigmoid squash the output [0, 1], and are used if your output is binary (not used in the hidden layers)
# 4. Tanh is similar to sigmoid, but squashes in [-1, 1]. It is rarely used any more.
# 5. Softmax normalizes the the output to 1, and is used as output if you have a classification problem
# 
# Change the current function to another. To do so, you must modify the forward()-method in the Net()-class. 
# 
# ### Early stopping
# Early stopping stops your training when you have reached the best possible model before overfitting. The method saves the model weights at each epoch while constantly monitoring the development of the validation loss. Once the validation loss starts to increase the method will raise a flag. The method will allow for a number of epochs to pass before stopping. The number of epochs are referred to as patience. If the validation loss decreases below the previous global minima before the patience runs out the flag and patience is reset. If a new global minima is not encountered the training is stopped and the weights from the global minima epoch are loaded and defines the final model. 
# 
# To implement early stopping you must set implement=True in the invoke()-function called within train() or train_with_minibatches().
# 
# ### Regularization
# Implement either L2 regularization, [dropout](https://pytorch.org/docs/stable/nn.html#dropout-layers) or [batch normalization](https://pytorch.org/docs/stable/nn.html#normalization-layers).
# 
# ## ... and you're done!

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# In[2]:


from pytorchtools import EarlyStopping


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from preprocessing_clone import *



# In[4]:


def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]


# In[5]:
'''

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
    return df[df.peptide.apply(len) <= MAX_PEP_SEQ_LEN]
'''

# In[6]:


def encode_peptides(Xin):
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


# In[7]:


def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True


# ## Arguments

# In[8]:


#MAX_PEP_SEQ_LEN = 9
BINDER_THRESHOLD = 0.426


# # Main

# ## Load

# In[9]:


blosum_file = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/BLOSUM50"
#train_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/A0201/f000"
#valid_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/A0201/f001"
#test_data = "C:/Users/white/Google Drive/Master's/June 2020/Algorithms in Bioinformatics/Algo/data/A0201/c000"


# In[10]:


#train_raw = load_peptide_target(train_data)
valid_raw = valid_data
test_raw = test_data


# ### Visualize Data

# In[11]:


def plot_target_values(data=[(train_raw, 'Train set'), (valid_raw, 'Validation set'), (test_raw, 'Test set')]):
    plt.figure(figsize=(15,4))
    for partition, label in data:
        x = partition.index
        y = partition.target
        plt.scatter(x, y, label=label, marker='.')
    plt.axhline(y=BINDER_THRESHOLD, color='r', linestyle='--', label='Binder threshold')
    plt.legend(frameon=False)
    plt.title('Target values')
    plt.xlabel('Index of dependent variable')
    plt.ylabel('Dependent varible')
    plt.show()


# In[12]:


plot_target_values()


# ### Encode data

# In[13]:


x_train_, y_train_ = encode_peptides(train_raw)
x_valid_, y_valid_ = encode_peptides(valid_raw)
x_test_, y_test_ = encode_peptides(test_raw)


# Check the data dimensions for the train set and validation set (batch_size, MAX_PEP_SEQ_LEN, n_features)

# In[14]:


print(x_train_.shape)
print(x_valid_.shape)
print(x_test_.shape)


# ### Flatten tensors

# In[15]:


x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)


# In[16]:


batch_size = x_train_.shape[0]
n_features = x_train_.shape[1]


# ### Make data iterable

# In[17]:


x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)

x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)

x_test = Variable(torch.from_numpy(x_test_.astype('float32')))
y_test = Variable(torch.from_numpy(y_test_.astype('float32'))).view(-1, 1)


# ## Build Model

# In[18]:


class Net(nn.Module):

    def __init__(self, n_features, n_l1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_l1)
        self.fc2 = nn.Linear(n_l1, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ## Select Hyper-parameters

# In[19]:


def init_weights(m):
    """
    https://pytorch.org/docs/master/nn.init.html
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0) # alternative command: m.bias.data.fill_(0.01)


# In[20]:


EPOCHS = 10000
MINI_BATCH_SIZE = 100
N_HIDDEN_NEURONS = 10
LEARNING_RATE = 0.01
PATIENCE = EPOCHS // 10


# ## Compile Model

# In[21]:


net = Net(n_features, N_HIDDEN_NEURONS)
#net.apply(init_weights)

net


# In[23]:


optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()


# ## Train Model

# In[24]:


# No mini-batch loading
def train():
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        net.train()
        pred = net(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data)

        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))

        net.eval()
        pred = net(x_valid)
        loss = criterion(pred, y_valid)  
        valid_loss.append(loss.data)

        if invoke(early_stopping, valid_loss[-1], net, implement=False):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss


# In[25]:


# Train with mini_batches
train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=TensorDataset(x_valid, y_valid), batch_size=MINI_BATCH_SIZE, shuffle=True)

def train_with_minibatches():
    
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)
    for epoch in range(EPOCHS):
        batch_loss = 0
        net.train()
        for x_train, y_train in train_loader:
            pred = net(x_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.data
        train_loss.append(batch_loss / len(train_loader))

        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss[-1]))

        batch_loss = 0
        net.eval()
        for x_valid, y_valid in valid_loader:
            pred = net(x_valid)
            loss = criterion(pred, y_valid)
            batch_loss += loss.data
        valid_loss.append(batch_loss / len(valid_loader))

        if invoke(early_stopping, valid_loss[-1], net, implement=False):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss


# In[26]:


net, train_loss, valid_loss = train()


# In[27]:


#net, train_loss, valid_loss = train_with_minibatches()


# In[28]:


def plot_losses(burn_in=20):
    plt.figure(figsize=(15,4))
    plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
    plt.plot(list(range(burn_in, len(valid_loss))), valid_loss[burn_in:], label='Validation loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Minimum Validation Loss')

    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
plot_losses()


# ## Evaluation

# ### Predict on test set

# In[29]:


net.eval()
pred = net(x_test)
loss = criterion(pred, y_test)


# In[30]:


plot_target_values(data=[(pd.DataFrame(pred.data.numpy(), columns=['target']), 'Prediction'),
                         (test_raw, 'Target')])


# ### Transform targets to class

# In[31]:


y_test_class = np.where(y_test.flatten() >= BINDER_THRESHOLD, 1, 0)
y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)


# ### Receiver Operating Caracteristic (ROC) curve

# In[32]:


fpr, tpr, threshold = roc_curve(y_test_class, pred.flatten().detach().numpy())
roc_auc = auc(fpr, tpr)


# In[33]:


def plot_roc_curve():
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_roc_curve()


# ### Matthew's Correlation Coefficient (MCC)

# In[34]:


mcc = matthews_corrcoef(y_test_class, y_pred_class)


# In[35]:


def plot_mcc():
    plt.title('Matthews Correlation Coefficient')
    plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'MCC = %0.2f' % mcc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.show()

plot_mcc()


# In[ ]:




