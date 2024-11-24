import torch
import os
import csv
import numpy as np
from aeon.datasets import load_from_tsv_file
from tqdm import tqdm
from dtaidistance import dtw
from copy import deepcopy
import math
import torch.nn.functional as F
import json
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,ChebConv,ARMAConv,SGConv,TAGConv
from torch.optim import Adam
from torch.nn import LeakyReLU
from torch.nn import Sigmoid
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import matplotlib.pyplot as 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, Parameter
from torch_geometric.nn import GCNConv, InnerProductDecoder  # Example layers; import others as needed
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
from sklearn.mixture import GaussianMixture as GMM
import itertools
import random

# Set the device to GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Optional: Ensure that only GPU 0 is visible to PyTorch
# torch.cuda.set_device(0)
device
class UCR_clustring():
  def __init__(self,dir,name):
    self.dir = dir
    self.name = name

  def load(self,clusting=True):
    path_train = os.path.join(self.dir,self.name,self.name+'_TRAIN.tsv')
    path_test = os.path.join(self.dir,self.name,self.name+'_TEST.tsv')
    X_train, y_train = load_from_tsv_file(path_train)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[-1])
    X_test, y_test = load_from_tsv_file(path_test)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[-1])

    if not clusting:
      return X_train, y_train,X_test, y_test
    X = np.concatenate([X_train,X_test])
    y = np.concatenate([y_train,y_test])
    self.X = X
    self.y = y
    return X ,y

  def DTW(self,window=20,penalty=0,derivative=False):
    if derivative:
      X = np.gradient(self.X,axis=1)
    else:
      X = self.X
    n = X.shape[0]
    dist_mat = np.zeros((n,n))
    for i,ser1 in (enumerate(X)):
      for j,ser2 in enumerate(X):
        if i<j:
          dist = dtw.distance_fast(X[i], X[j],window=window,penalty=penalty)
          dist_mat[i,j] = dist
          dist_mat[j,i] = dist
    self.dist_mat = dist_mat
    return dist_mat

  def graph_exp(self,dist_mat=None,trshs=None):
    if not dist_mat:
      dist_mat = self.dist_mat
    if not trshs:
      trshs = [0.1,0.5,1,4]
    graph = np.zeros((len(trshs),dist_mat.shape[0],dist_mat.shape[1]))
    for i,trsh in enumerate(trshs):
       graph[i] = np.exp(-trsh*dist_mat)
    return graph

  def sparse_graph(self,rate,dist_mat=None):
    if not dist_mat:
      dist_mat = deepcopy(self.dist_mat)
    sgraph = np.ones_like(dist_mat)
    shape = dist_mat.shape
    n = shape[1]*shape[0]
    out,count = np.unique(dist_mat,return_counts=True)
    i = 0
    j = 0
    trsh = 0
    while i/n<rate:
      i+=count[j]
      trsh = out[j]
      j+=1
    sgraph[dist_mat>=trsh]=0
    return sgraph


def load_data_with_labels_ucr(dataset_str, sparcity,window=None,penalty=None,derivative=False):
    ucr = UCR_clustring('/content/drive/MyDrive/DataSet/UCRArchive',dataset_str)
    x,y = ucr.load()
    y -=1
    num_classes = len(np.unique(y))

    dtw_mat = ucr.DTW(window,penalty,derivative)
    graph = ucr.graph_exp()
    adj = ucr.sparse_graph(sparcity)

    x = torch.tensor(x)

    # Convert the dense matrix to a sparse matrix of type '<class 'numpy.int64'>'
    # adj = csr_matrix(sgraph, dtype=np.int64)
    list_1 = []
    list_2 = []
    for i in range(np.shape(adj)[0]):
        for j in range(np.shape(adj)[1]):
            if(adj[i,j]!=0):
              list_1.append(i)
              list_2.append(j)
    edge_list = [list_1,list_2]

    # edge_index = np.array(edge_list,dtype=int)
    edge_index = torch.tensor(edge_list).long().to(device)

    min_val = x.min(dim=0, keepdim=True)[0]
    max_val = x.max(dim=0, keepdim=True)[0]
    normalized_features = (x - min_val) / (max_val - min_val)
    X = normalized_features.float().to(device)

    adj = torch.tensor(adj).float().to(device)

    return adj, edge_index, X, y , num_classes,ucr
def update_graph(ucr,sparsity):
    x,y = ucr.X,ucr.y
    y -=1
    num_classes = len(np.unique(y))
    adj = ucr.sparse_graph(sparsity)

    x = torch.tensor(x)

    # Convert the dense matrix to a sparse matrix of type '<class 'numpy.int64'>'
    # adj = csr_matrix(sgraph, dtype=np.int64)
    list_1 = []
    list_2 = []
    for i in range(np.shape(adj)[0]):
        for j in range(np.shape(adj)[1]):
            if(adj[i,j]!=0):
              list_1.append(i)
              list_2.append(j)
    edge_list = [list_1,list_2]

    # edge_index = np.array(edge_list,dtype=int)
    edge_index = torch.tensor(edge_list).long().to(device)

    min_val = x.min(dim=0, keepdim=True)[0]
    max_val = x.max(dim=0, keepdim=True)[0]
    normalized_features = (x - min_val) / (max_val - min_val)
    X = normalized_features.float().to(device)

    adj = torch.tensor(adj).float().to(device)


    return adj, edge_index, X, y , num_classes,ucr

