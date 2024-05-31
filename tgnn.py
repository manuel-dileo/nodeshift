import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import random

import copy

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear

import torch
import networkx as nx
import numpy as np

from sklearn.metrics import *


import pandas as pd
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear

import matplotlib.pyplot as plt

import itertools
import json

class ROLANDLP(torch.nn.Module):
    def __init__(self, input_dim, num_nodes, dropout=0.0, update='mlp', loss=BCEWithLogitsLoss):
        
        super(ROLANDLP, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess node repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        #You can change the layer dimensions but 
        #if you change the architecture you need to change the forward method too
        #TODO: make the architecture parameterizable
        
        hidden_conv_1 = 64 
        hidden_conv_2 = 32
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        self.conv1 = GCNConv(128, hidden_conv_1)
        self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2)
        self.postprocess1 = Linear(hidden_conv_2, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.dropout = dropout
        self.update = update
        
        self.tau0 = torch.nn.Parameter(torch.Tensor([0.2]))
        if update=='gru':
            self.gru1 = GRUCell(hidden_conv_1, hidden_conv_1)
            self.gru2 = GRUCell(hidden_conv_2, hidden_conv_2)
        elif update=='mlp':
            self.mlp1 = Linear(hidden_conv_1*2, hidden_conv_1)
            self.mlp2 = Linear(hidden_conv_2*2, hidden_conv_2)
        self.previous_embeddings = None
                                    
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()
        

    def forward(self, x, edge_index, edge_label_index=None, isnap=0, previous_embeddings=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None and isnap > 0: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone(),previous_embeddings[1].clone()]
        
        current_embeddings = [torch.Tensor([]),torch.Tensor([])]
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        h = F.dropout(h, p=self.dropout,inplace=True)
        h = self.preprocess2(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, inplace=True)
        
        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, inplace=True)
        #Embedding Update after first layer
        if isnap > 0:
            if self.update=='gru':
                h = torch.Tensor(self.gru1(h, self.previous_embeddings[0].clone()).detach().numpy())
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[0].clone()),dim=1)
                h = torch.Tensor(self.mlp1(hin).detach().numpy())
            else:
                h = torch.Tensor((self.tau0 * self.previous_embeddings[0].clone() + (1-self.tau0) * h.clone()).detach().numpy())
       
        current_embeddings[0] = h.clone()
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, inplace=True)
        #Embedding Update after second layer
        if isnap > 0:
            if self.update=='gru':
                h = torch.Tensor(self.gru2(h, self.previous_embeddings[1].clone()).detach().numpy())
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[1].clone()),dim=1)
                h = torch.Tensor(self.mlp2(hin).detach().numpy())
            else:
                h = torch.Tensor((self.tau0 * self.previous_embeddings[1].clone() + (1-self.tau0) * h.clone()).detach().numpy())
      
        current_embeddings[1] = h.clone()
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.postprocess1(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        #return both 
        #i)the predictions for the current snapshot 
        #ii) the embeddings of current snapshot

        return h, current_embeddings
    
    def tau(self):
        if self.update=='lwa':
            return self.tau0
        raise Exception('update!=lwa')
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
    
def test(model, test_data, data, isnap, device='cpu'):
    model.eval()

    test_data = test_data.to(device)

    h, _ = model(test_data.x, test_data.edge_index, edge_label_index = test_data.edge_label_index, isnap=isnap)
    
    pred_cont_link = torch.sigmoid(h).cpu().detach().numpy()
    
    label_link = test_data.edge_label.cpu().detach().numpy()
      
    avgpr_score_link = average_precision_score(label_link, pred_cont_link)
    
    return avgpr_score_link

def train_single_snapshot(model, data, train_data, val_data, test_data, isnap,\
                          last_embeddings, optimizer, device='cpu', num_epochs=50, verbose=False):
    
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_current_embeddings = []
    
    avgpr_trains = []
    #avgpr_vals = []
    avgpr_tests = []
    
    tol = 1
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred,\
        current_embeddings =\
            model(train_data.x, train_data.edge_index, edge_label_index = train_data.edge_label_index,\
                  isnap=isnap, previous_embeddings=last_embeddings)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val  = test(model, val_data, data, isnap, device)
        
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = model
        else:
            break
        
        
    avgpr_score_train = test(model, train_data, data, isnap, device)
    avgpr_score_test = test(model, test_data, data, isnap, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, optimizer, avgpr_score_train, avgpr_score_test, best_current_embeddings

def train_roland(snapshots, hidden_conv1, hidden_conv2, device='cpu'):
    num_snap = len(snapshots)
    input_channels = snapshots[0].x.size(1)
    num_nodes = snapshots[0].x.size(0)
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_conv1)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_conv2)] for j in range(num_nodes)])]
    avgpr_train_singles = []
    avgpr_test_singles = []
    mrr_train_singles = []
    mrr_test_singles = []
    
    roland = ROLANDLP(input_channels, num_nodes, update='gru')
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=0.01, weight_decay = 5e-3)
    roland.reset_parameters()
    
    node_states = {}
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1])
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)

        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        roland, rolopt, avgpr_train, avgpr_test, last_embeddings =\
            train_single_snapshot(roland, snapshot, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt)
        
        node_states[i] = last_embeddings
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tLinkPre AVGPR Train: {avgpr_train}, Test: {avgpr_test}')
        avgpr_train_singles.append(avgpr_train)
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_train_all = sum(avgpr_train_singles)/len(avgpr_train_singles)
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    #obtain the node embeddings on the last snapshot (test_data at i=3)
    _, last_embeddings = roland(test_data.x, test_data.edge_index, test_data.edge_label_index, num_snap-1, last_embeddings)
    node_states[num_snap-1] = last_embeddings
    
    print(f'LinkPre AVGPR over time: Train {avgpr_train_all}, Test: {avgpr_test_all}')
    
    return roland, node_states