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
import matplotlib.pyplot as plt
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
class Encoder(nn.Module):
    """
    Encoder module with two convolutional layers.

    Args:
        in_channels (int): Number of input features.
        hidden1_channels (int): Number of features in the first hidden layer.
        hidden2_channels (int): Number of features in the second hidden layer.
        conv_kwargs (dict, optional): Additional arguments for the convolutional layers.
    """
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, Conv=GCNConv, conv_kwargs=None):
        super(Encoder, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.gcn_shared1 = Conv(in_channels, hidden1_channels, **conv_kwargs)
        self.relu1 = LeakyReLU(0.2)
        self.gcn_shared2 = Conv(hidden1_channels, hidden2_channels, **conv_kwargs)
        self.relu2 = LeakyReLU(0.2)

    def forward(self, X, Adj):
        Z = self.gcn_shared1(X, Adj)
        Z = self.relu1(Z)
        Z = self.gcn_shared2(Z, Adj)
        Z = self.relu2(Z)
        return Z


class InnerProductDecoder(nn.Module):
    """
    Decoder that uses the inner product for prediction.

    Args:
        dropout (float): Dropout rate.
        act (torch.nn.Module): Activation function, default is torch.sigmoid.
    """
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class Decoder_X(nn.Module):
    """
    Decoder module with three convolutional layers.

    Args:
        latent_channels (int): Number of input features from the latent space.
        hidden1_channels (int): Number of features in the first hidden layer.
        hidden2_channels (int): Number of features in the second hidden layer.
        out_channels (int): Number of output features.
        conv_kwargs (dict, optional): Additional arguments for the convolutional layers.
    """
    def __init__(self, latent_channels, hidden1_channels, hidden2_channels, out_channels,
                 Conv=GCNConv, conv_kwargs=None):
        super(Decoder_X, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.gcn1 = Conv(latent_channels, hidden1_channels, **conv_kwargs)
        self.relu1 = LeakyReLU(0.2)
        self.gcn2 = Conv(hidden1_channels, hidden2_channels, **conv_kwargs)
        self.relu2 = LeakyReLU(0.2)
        self.gcn3 = Conv(hidden2_channels, out_channels, **conv_kwargs)
        self.relu3 = LeakyReLU(0.2)

    def forward(self, Z, Adj):
        Z = self.gcn1(Z, Adj)
        Z = self.relu1(Z)
        Z = self.gcn2(Z, Adj)
        Z = self.relu2(Z)
        Z = self.gcn3(Z, Adj)
        X_rec = self.relu3(Z)
        return X_rec


class GVADE_For_Pretrain(nn.Module):
    """
    Auto-Encoder for pretraining VaDE.

    Args:
        in_channels (int): Number of input features.
        hidden1_channels (int): Number of features in the first hidden layer.
        hidden2_channels (int): Number of features in the second hidden layer.
        latent_channels (int): Number of features in the latent space.
        n_classes (int): Number of clusters/classes.
        dropout (float, optional): Dropout rate for the decoder.
        conv_kwargs (dict, optional): Additional arguments for the convolutional layers.
    """
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, latent_channels,
                 n_classes, dropout=0.4, Conv=GCNConv, conv_kwargs=None):
        super(GVADE_For_Pretrain, self).__init__()

        self.encoder = Encoder(in_channels, hidden1_channels, hidden2_channels,Conv, conv_kwargs)
        self.encoder_mu = Conv(hidden2_channels, latent_channels, **(conv_kwargs or {}))
        self.tanh = nn.Tanh()

        self.decoder = InnerProductDecoder(dropout)

    def encode(self, X, A):
        x = self.encoder_mu(self.encoder(X, A), A)
        return x  # self.tanh(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X, A):
        z = self.encode(X, A)
        recon_x = self.decode(z)
        return recon_x


class GVADE(nn.Module):
    """
    Variational Auto-Encoder (VaDE) with graph convolutional layers.

    Args:
        in_channels (int): Number of input features.
        hidden1_channels (int): Number of features in the first hidden layer.
        hidden2_channels (int): Number of features in the second hidden layer.
        latent_channels (int): Number of features in the latent space.
        n_classes (int): Number of clusters/classes.
        dropout (float, optional): Dropout rate for the decoder.
        conv_kwargs (dict, optional): Additional arguments for the convolutional layers.
    """
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, latent_channels,
                 n_classes, dropout=0.1, Conv=GCNConv, conv_kwargs=None):
        super(GVADE, self).__init__()

        self.encoder = Encoder(in_channels, hidden1_channels, hidden2_channels,Conv ,conv_kwargs)

        self.encoder_mu = Conv(hidden2_channels, latent_channels, **(conv_kwargs or {}))
        self.encoder_logvar = Conv(hidden2_channels, latent_channels, **(conv_kwargs or {}))
        self.tanh = nn.Tanh()

        # self._pi = Parameter(torch.zeros(n_classes))
        # self.mu = Parameter(torch.randn(n_classes, latent_channels))
        # self.logvar = Parameter(torch.randn(n_classes, latent_channels))

        self._pi=nn.Parameter(torch.FloatTensor(n_classes,).fill_(1)/n_classes,requires_grad=True)
        self.mu=nn.Parameter(torch.FloatTensor(n_classes,latent_channels).fill_(0),requires_grad=True)
        self.logvar=nn.Parameter(torch.FloatTensor(n_classes,latent_channels).fill_(0),requires_grad=True)
        self.n_classes = n_classes

        self.decoder = InnerProductDecoder(dropout)

    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu).to(mu.device)
        else:
            return mu.to(mu.device)

    def encode(self, X, A):
        h = self.encoder(X, A)
        mu = self.encoder_mu(h, A)  # self.tanh(self.encoder_mu(h, A))
        logvar = self.encoder_logvar(h, A)  # self.tanh(self.encoder_logvar(h, A))
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X, A):
        mu, logvar = self.encode(X, A)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

bce_loss = torch.nn.BCELoss()
def pretraining(gvae_for_pretrain,optimizer,max_epoch, X, A, edge_indeces):
    # Set early stopping parameters
    early_stopping_patience = 60  # Number of epochs to wait before stopping if no improvement
    best_loss = float('inf')  # Initialize the best loss as infinity
    patience_counter = 0  # Counter for early stopping
    gvae_for_pretrain.train()
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        A_rec = gvae_for_pretrain.forward(X, edge_indeces)

        # Calculate the loss
        loss = bce_loss(A_rec, A)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        # Check if the current loss is the best we've seen so far
        if loss < best_loss:
            best_loss = loss  # Update the best loss
            patience_counter = 0  # Reset the counter since we got an improvement
        else:
            patience_counter += 1  # Increment the counter if no improvement

        # Early stopping condition
        if patience_counter >= early_stopping_patience:
            # print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    return gvae_for_pretrain
    # Optionally, print the progress
    # print(f"Epoch {epoch + 1}/{50}, Loss: {loss.item()}")


def set_model(gvae_for_pretrain, model, X,edge_indeces, num_classes):
    with torch.no_grad():
        z = gvae_for_pretrain.encode(X,edge_indeces).cpu()

    # print(f"z.shape: {z.shape}")
    gmm = GaussianMixture(n_components=num_classes, covariance_type='diag')
    gmm.fit(z)

    gvae_for_pretrain.cpu()
    state_dict = gvae_for_pretrain.state_dict()

    model.load_state_dict(state_dict, strict=False)
    model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
    model.mu.data = torch.from_numpy(gmm.means_).float()
    model.logvar.data = (torch.log(torch.from_numpy(gmm.covariances_)).float()) / 10.0
    model.to(device)
    return model
def loss_func(mu,logvar,model,n_nodes):
        z = model.reparameterize(mu, logvar).unsqueeze(1)
        h = z - model.mu

        h = h.double()

        h = torch.exp(-0.5 * torch.sum((h * h / (model.logvar.exp())), dim=2))

        h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
        p_z_given_c = h / (2 * math.pi)



        p_z_c = p_z_given_c * model.weights

        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

        h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
        h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
        loss = 0.5 * torch.sum(gamma * h) - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
              + torch.sum(gamma * torch.log(gamma + 1e-9)) \
              - 0.5 * torch.sum(1 + logvar)

        loss = loss / n_nodes

        if torch.isnan(loss):
            loss = 0.0 # torch.tensor(0.0, requires_grad=True)


        return loss
def gaussian_pdfs_log(x,mus,log_sigma2s,nClusters):
    G=[]
    for c in range(nClusters):
        G.append(gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
    return torch.cat(G,1)

def gaussian_pdf_log(x,mu,log_sigma2):
    return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))
def ELBO_Loss(z_mu,z_sigma2_log,model,n_nodes,L=1):
    det=1e-10

    L_rec=0

    pi=model.weights
    log_sigma2_c = model.logvar
    mu_c=model.mu

    z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
    yita_c=torch.exp(torch.log(pi.unsqueeze(0))+gaussian_pdfs_log(z,mu_c,log_sigma2_c,model.n_classes))+det

    yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

    Loss=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                            torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                            (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

    Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


    return Loss/n_nodes
def training(model,optimizer,max_epoch, X, A,edge_indeces, num_classes, landa):
    # Set early stopping parameters
    early_stopping_patience = 60  # Number of epochs to wait before stopping if no improvement
    best_loss = float('inf')  # Initialize the best loss as infinity
    patience_counter = 0  # Counter for early stopping
    print('Training mode : ')
    model.train()
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        A_rec, mean, log_var = model.forward(X, edge_indeces)

        # Calculate the loss
        loss = bce_loss(A_rec, A)
        secloss = loss_func(mean, log_var, model, X.shape[0])

        loss += landa * secloss

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        # Check if the current loss is the best we've seen so far
        if loss < best_loss:
            best_loss = loss  # Update the best loss
            patience_counter = 0  # Reset the counter since we got an improvement
        else:
            patience_counter += 1  # Increment the counter if no improvement

        # Early stopping condition
        #if patience_counter >= early_stopping_patience:
            # print('Early stopped!')
            # print(f"Early stopping triggered after {epoch + 1} epochs.")
            #break

    return model
def evaluate(Y,mean, num_classes,seed):
    # Initialize lists to store metrics

    results = []

    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    gmm = GMM(n_components=num_classes, random_state=seed, n_init=10)

    y_pred_kmeans = kmeans.fit_predict(mean.detach().cpu().numpy())
    y_pred_gmm = gmm.fit_predict(mean.detach().cpu().numpy())

    # Calculate metrics
    nmi_kmeans = normalized_mutual_info_score(Y, y_pred_kmeans)
    ri_kmeans = rand_score(Y, y_pred_kmeans)
    sil_kmeans = np.float64(silhouette_score(mean.detach().cpu().numpy(),y_pred_kmeans))
    dbi_kmeans = davies_bouldin_score(mean.detach().cpu().numpy(),y_pred_kmeans)
    ch_kmeans = calinski_harabasz_score(mean.detach().cpu().numpy(),y_pred_kmeans)
    nmi_gmm = normalized_mutual_info_score(Y, y_pred_gmm)
    ri_gmm = rand_score(Y, y_pred_gmm)
    sil_gmm = np.float64(silhouette_score(mean.detach().cpu().numpy(),y_pred_gmm))
    dbi_gmm = davies_bouldin_score(mean.detach().cpu().numpy(),y_pred_gmm)
    ch_gmm = calinski_harabasz_score(mean.detach().cpu().numpy(),y_pred_gmm)
    # Store results in a dictionary
    result = {
        "seed": seed,
        "nmi_kmeans": nmi_kmeans,
        "ri_kmeans": ri_kmeans,
        'sil_kmeans' : sil_kmeans,
        'dbi_kmeans' : dbi_kmeans,
        'ch_kmeans' : ch_kmeans,
        'nmi_gmm' : nmi_gmm,
        'ri_gmm' : ri_gmm,
        'sil_gmm' : sil_gmm,
        'dbi_gmm' : dbi_gmm,
        'ch_gmm' : ch_gmm
    }


    return result