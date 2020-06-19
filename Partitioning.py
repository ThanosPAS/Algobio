
import numpy as np
from time import time
from argparse import ArgumentParser
import pickle

parser = ArgumentParser(description="Compute redundancy and create data partitions (without disrupting cluster structures)")
parser.add_argument("-wd", action="store", dest="data_dir", type=str, help="Path to cluster file")
parser.add_argument("-sd", action="store", dest="save_dir", type=str, help="Path for saving partition files")
parser.add_argument("-k", action="store", dest="k_fold", type=int, default=5, help="Choose k-fold CV. Provide int. Default: 5")
parser.add_argument("-ts", action="store", dest="test_size", type=float, default=0.33, help="Provide decimal for size of final evaluation test set. Default: 0.33")
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir

k_fold = args.k_fold
test_size = args.test_size

with open(data_dir + "clusters.txt", "rb") as new_filename:
    accepted_clusters = pickle.load(new_filename)


# In[73]:


sorted_clusters = sorted(accepted_clusters, key=len, reverse = True)


# In[74]:


def getSizeOfNestedList(listOfElem):
    count = 0
    for elem in listOfElem:
        if type(elem) == list:
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count


# In[75]:


fold = k_fold
test_to_train_size = test_size
partitions = fold + round(fold * test_to_train_size)
buckets = [[] for i in range(partitions)]

for cluster in sorted_clusters:
    size_of_buckets = []
    for i in range(partitions):
        size_of_buckets.append(getSizeOfNestedList(buckets[i]))
    min_pos = size_of_buckets.index(min(size_of_buckets))
    buckets[min_pos].append(cluster)


# In[76]:


merged_list = [[] for i in range(partitions)]
index = 0
for sublist in buckets:
    for subsublist in sublist:
        merged_list[index] += subsublist
    index +=1


# In[77]:


for i in range(len(merged_list)):
    print('size of partition', i,": ", len(merged_list[i]))


# In[78]:


for i in range(len(merged_list)):
    with open(save_dir + "partition_" + str(i) + ".txt", "wb") as internal_filename:
        pickle.dump(merged_list[i], internal_filename)
