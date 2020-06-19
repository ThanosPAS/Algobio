#!/usr/bin/env python
# coding: utf-8

# # Hobohm 1
#
# ### Fill in the blanked out part of the code (XX)

# ## Python Imports

# In[1]:


import numpy as np
from time import time
from argparse import ArgumentParser


# ## Data Imports

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[2]:


#data_dir = "C:/Users/vinde/OneDrive/Dokumenter/DTU/Algorithms_in_bioinformatics/Project/Cluster_data/"
#data_dir_2 = "C:/Users/vinde/OneDrive/Dokumenter/DTU/Algorithms_in_bioinformatics/data/"


# In[ ]:


parser = ArgumentParser(description="Compute redundancy and create data partitions (without disrupting cluster structures)")
parser.add_argument("-wd1", action="store", dest="data_dir", type=str, help="Path to data file (A0201.txt) - make sure to use the .txt uploaded to repository")
parser.add_argument("-wd2", action="store", dest="data_dir_2", type=str, help="Path to alphabet and BLOSUM matrix")
parser.add_argument("-sd", action="store", dest="save_dir", type=str, help="Path for saving partition files")
parser.add_argument("-hf", action="store", dest="threshold", type=int, default=7, help="Choose appropriate homology function (number of matches between peptides to compute redundancy - for 9mer peptides default: 7)")


args = parser.parse_args()

data_dir = args.data_dir
data_dir_2 = args.data_dir_2
save_dir = args.save_dir
threshold = args.threshold



# In[3]:


alphabet_file = data_dir_2 + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

alphabet

blosum_file = data_dir_2 + "Matrices/BLOSUM50"
_blosum50 = np.loadtxt(blosum_file, dtype=int).T

blosum50 = {}

for i, letter_1 in enumerate(alphabet):

    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):

        blosum50[letter_1][letter_2] = _blosum50[i, j]

#blosum50


# ### Sequences

# In[4]:


def load_sequences():

    #database_file = data_dir_2 + "Hobohm/database_list.tab"
    #database_list = np.loadtxt(database_file, dtype=str).reshape(-1,2)

    database_file = data_dir + "A0201.txt"
    database_list = np.loadtxt(database_file, dtype=str).reshape(-1,2)

    #ids = database_list[:, 0]
    #sequences = database_list[:, 1]

    ids = database_list[:, 0]
    sequences = database_list[:, 0]

    return sequences, ids


# ## Smith-Waterman O2
#
# ### This code is identical to the code you wrote the other day

# In[5]:


def smith_waterman(query, database, scoring_scheme, gap_open, gap_extension):

    P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)

    aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)

    return aligned_query, aligned_database, matches


def smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension):

    # Matrix imensions
    M = len(query)
    N = len(database)

    # D matrix change to float
    D_matrix = np.zeros((M+1, N+1), np.int)

    # P matrix
    P_matrix = np.zeros((M+1, N+1), np.int)

    # Q matrix
    Q_matrix = np.zeros((M+1, N+1), np.int)

    # E matrix
    E_matrix = np.zeros((M+1, N+1), dtype=object)

    # Main loop
    D_matrix_max_score, D_matrix_i_max, D_matrix_i_max = -9, -9, -9
    for i in range(M-1, -1, -1):
        for j in range(N-1, -1, -1):

            # Q_matrix[i,j] entry
            gap_open_database = D_matrix[i+1,j] + gap_open
            gap_extension_database = Q_matrix[i+1,j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)

            Q_matrix[i,j] = max_gap_database

            # P_matrix[i,j] entry
            gap_open_query = D_matrix[i,j+1] + gap_open
            gap_extension_query = P_matrix[i,j+1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)

            P_matrix[i,j] = max_gap_query

            # D_matrix[i,j] entry
            diagonal_score = D_matrix[i+1,j+1] + scoring_scheme[query[i]][database[j]]

            # E_matrix[i,j] entry
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]

            direction, max_score = max(candidates, key=lambda x: x[1])


            # check entry sign
            if max_score > 0:
                E_matrix[i,j] = direction
            else:
                E_matrix[i,j] = 0

            # check max score sign
            if max_score > 0:
                D_matrix[i, j] = max_score
            else:
                D_matrix[i, j] = 0

            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j

    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension):

    # Matrix imensions
    M = len(query)
    N = len(database)

    # aligned query string
    aligned_query = []

    # aligned database string
    aligned_database = []

    # total identical matches
    matches = 0


    # start from max_i, max_j
    i, j = i_max, j_max
    while i < M and j < N:

        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break

        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if ( query[i] == database[j]):
                matches += 1
            i += 1
            j += 1


        # E[i,j] = 2, gap opening in database
        if E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1


        # E[i,j] = 3, gap extension in database
        if E_matrix[i, j] == 3:

            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension

            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.00001):
                count += 1
                score = D_matrix[count, j] + gap_open + (count-i-1)*gap_extension

            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1


        # E[i,j] = 4, gap opening in query
        if E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1


        # E[i,j] = 5, gap extension in query
        if E_matrix[i, j] == 5:

            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension

            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.0001):
                count += 1
                score = D_matrix[i, count] + gap_open + (count-j-1)*gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1


    return aligned_query, aligned_database, matches


# ## Hobohm 1

# ### Similarity Function
#
# ### This code defines the threshold for similarity
#
# ### Homology score for 9-mer peptides: mismatch > 2/9

# In[6]:


def homology_function(alignment_length, matches, min_matches):

    homology_score = min_matches
    #homology_score = XX, this was just Ole's formula
    #homology_score = 2.9*np.sqrt(alignment_length)**0.5

    #if matches XX homology_score: ## Add the inequally sign
    if matches > homology_score:
        return "discard", homology_score
    else:
        return "keep", homology_score


# ### Main Loop Initiation

# In[7]:


# load list
candidate_sequences, candidate_ids = load_sequences()
print ("# Numner of elements:", len(candidate_sequences))

accepted_sequences, accepted_ids = [], []

accepted_sequences.append(candidate_sequences[0])
accepted_ids.append([candidate_ids[0]])
#print(accepted_ids)
print("# Unique.", 0, len(accepted_sequences)-1, accepted_ids[0][0])


# ### Main Loop

# In[8]:


# parameters
scoring_scheme = blosum50
gap_open = -11
gap_extension = -1

t0 = time()

for i in range(1, len(candidate_sequences)):
#for i in range(1, 41):

    for j in range(0, len(accepted_sequences)):

        query = candidate_sequences[i]
        database = accepted_sequences[j]

        aligned_query, aligned_database, matches = smith_waterman(query, database, scoring_scheme, gap_open, gap_extension)

        alignment_length = len(aligned_query)

        homology_outcome, homology_score = homology_function(alignment_length, matches, threshold)

        # query is not unique
        if homology_outcome == "discard":

            accepted_ids[j].append(candidate_ids[i])
            print ("# Not unique.", i, candidate_ids[i], "is homolog to", accepted_ids[j][0], homology_score)

            break


    # query is unique
    if homology_outcome == "keep":
        #accepted_sequences.append(XX)
        accepted_sequences.append(candidate_sequences[i])
        #accepted_ids.append(XX)
        accepted_ids.append([candidate_ids[i]])

        print ("# Unique.", i, len(accepted_sequences)-1, candidate_ids[i], homology_score)

t1 = time()

print("Elapsed time (m):", (t1-t0)/60)

print ("Accepted clusters:", len(accepted_ids))
for i in range(len(accepted_ids)):
    print( accepted_ids[i])


# In[9]:


print("Elapsed time (m):", (t1-t0)/60)

print ("Accepted clusters:", len(accepted_ids))
for i in range(len(accepted_ids)):
    print( accepted_ids[i])


# In[71]:


import pickle
with open(save_dir + "clusters.txt", "wb") as internal_filename:
    pickle.dump(accepted_ids, internal_filename)


# In[72]:





# In[63]:
