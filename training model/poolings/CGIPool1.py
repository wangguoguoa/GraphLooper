from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv,Sequential
from torch_geometric.data import Data
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops, remove_self_loops
import torch.nn as nn
from torch_sparse import SparseTensor

import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_sparse import spspmm, coalesce
from torch_scatter import scatter_add, scatter
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import random
import scipy.sparse as sp
import numpy as np


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.sigmoid(self.fc2(x))
        return x
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels):
#         super(Encoder, self).__init__()
#         self.hidden_dim=in_channels
#         self.conv = GraphConv(in_channels,self.hidden_dim)
#         # print(self.conv)
#         self.fc3 = nn.Linear(self.hidden_dim ,self.hidden_dim)
#         self.fc4 = nn.Linear(self.hidden_dim, 1)
#
#     def forward(self,x,edge_index):
#         x = self.conv(x, edge_index)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         x =self.fc4(x)
#         return x
class Encoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x =self.fc2(x)
        return x


class Predictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(Predictor, self).__init__()

        self.p1 = nn.Linear(1, in_channels)
        self.p2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        print(x.shape)
        print(self.p1(x))
        x = F.leaky_relu(self.p1(x), 0.2)
        x =self.p2(x)
        return x

class CGIPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_lin=torch.tanh):
        super(CGIPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_lin = non_lin
        self.hidden_dim = in_channels

        self.transform = GraphConv(in_channels, self.hidden_dim)
        self.pp_conv = GraphConv(self.hidden_dim, self.hidden_dim)
        self.np_conv = GraphConv(self.hidden_dim, self.hidden_dim)

        self.positive_pooling = GraphConv(self.hidden_dim, 1) 
        self.negative_pooling = GraphConv(self.hidden_dim, 1)
        # self.encoder= Sequential('x, edge_index', [(GraphConv(in_channels* 2,  in_channels), 'x, edge_index -> x'),nn.LeakyReLU(inplace=True),(GraphConv(in_channels,self.hidden_dim), 'x, edge_index -> x'),nn.LeakyReLU(inplace=True),nn.Linear(self.hidden_dim, self.hidden_dim),nn.LeakyReLU(),nn.Linear(self.hidden_dim, 1)])
        # #self.predictor=Sequential(nn.Linear(in_channels * 2, in_channels),nn.LeakyReLU(),nn.Linear(in_channels, 1))
        # # self.gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
        # #             normalization_out='col',
        # #             diffusion_kwargs=dict(method='ppr', alpha=0.05),
        # #             sparsification_kwargs=dict(method='topk', k=128,
        # #                                        dim=0), exact=True)
        # #
        self.encoder1=GraphConv(1024, self.hidden_dim)
        self.encoder2=Encoder(self.hidden_dim)
        # self.hidden_dim1 = self.hidden_dim
        # # self.encoder=Encoder(self.hidden_dim1)
        self.predictor=Predictor(self.hidden_dim)
        self.discriminator = Discriminator(self.hidden_dim)
        self.loss_fn = nn.BCELoss()

    def permute_edges(self,data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num / 10)

        edge_index = data.edge_index.transpose(0, 1).cpu().numpy()

        idx_add = np.random.choice(node_num, (permute_num, 2))
        # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

        # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
        # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
        edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]
        # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

        return data
    def forward(self, x, edge_index, edge_attr=None, batch=None): #3662张图，每张图的特征维度为38，即3662x38,7958/2条边，batch为3662
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # da=np.ones(len(edge_index[0,:]))
        # le=len(edge_index[0,:])%3
        # print(da,le)
        # s = sp.csr_matrix(da,(edge_index[0,:],edge_index[1,:]))

        # adj = SparseTensor(row=edge_index[0], col=edge_index[1],sparse_sizes=(x.size(0), x.size(0)))
        # adj=torch.zeros(x.size(0),x.size(0))
        # print(edge_index[1,:].shape,edge_index[0,:].shape)
        # index1=torch.LongTensor(.cpu.numpy(),edge_index[0,:].cpu.numpy())
        # print(adj.scatter(1, index1, 1))

        # print(adj)

        # diff_data = self.gdc(diff_data)

        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2) #输入图的图表示 #3662X128
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)  #3662x128
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)

        s_pp = self.positive_pooling(x_tp, edge_index).squeeze() #score vector y l+1 #3662
        # print(s_pp.size(0))
        s_np = self.negative_pooling(x_tn, edge_index).squeeze() ##torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        # print("*****",s_pp.shape,s_np.shape)
        perm_positive = topk(s_pp,1, batch)# indices idx# per_postive 128
        perm_negative = topk(s_np, 1, batch)

        x_pp = x_transform[perm_positive] * self.non_lin(s_pp[perm_positive]).view(-1, 1) #the feature matrices H ##128x128

        x_np = x_transform[perm_negative] * self.non_lin(s_np[perm_negative]).view(-1, 1) # #哈玛达乘积
        f_pp, _ = filter_adj(edge_index, edge_attr, perm_positive, num_nodes=s_pp.size(0))
        # print(f_pp)

        x_pp_readout = gap(x_pp, batch[perm_positive]) # gap is global_mean_pool
        x_np_readout = gap(x_np, batch[perm_negative])
        x_readout = gap(x_transform, batch)

        positive_pair = torch.cat([x_pp_readout, x_readout], dim=1)#torch.cat 拼接 READOUT对topk后的节点进行一个全局平均池化和全局最大池化操作 128x256,128为batch_size，即选取了128个子图
        negative_pair = torch.cat([x_np_readout, x_readout], dim=1)  #特征拼接 128x256
        data = self.permute_edges(Data(positive_pair, edge_index)).cuda()
        # positive_pair1,edge_index1=data.x,data.edge_index
        # h1=F.leaky_relu(self.encoder1(data.x, data.edge_index), 0.2)     #encoder1 GraphConv
        # z1=self.encoder2(h1)
        # h2 = F.leaky_relu(self.encoder1(positive_pair1, edge_index1), 0.2)  # encoder1 GraphConv
        # z2 = self.encoder2(h2)
        # p1=self.predictor(z1)
        # p2 = self.predictor(z2)
        #aug_adj1=self.aug_random_edge(edge_index,0.4)  # #Edge Modification (EM) 边修改
        #print(positive_pair[0,:].shape,f_pp.shape)
        # edg_index1=edge_index.index_select(0,perm_positive)
        # print(edg_index1)

        # diff_data=gdc(diff_data)

        # aug_adj1 = self.aug_random_edge(edge_index)  # #Edge Modification (EM) 边修改
        # print(edge_index,aug_adj1)
        # print(positive_pair.size(1))
        # self.encoder = Encoder(positive_pair.size(1)).cuda()
        # print("###",positive_pair.shape,edge_index.shape)
        # z1 = self.encoder(positive_pair,edge_index )  # NxC
        # z2 = self.encoder(positive_pair1,edge_index1)  # NxC
        # print(z1)
        # p1 = self.predictor(z1)  # NxC
        # p2 = self.predictor(z2)  # NxC



        real = torch.ones(positive_pair.shape[0],1).cuda()  #128x1
        fake = torch.zeros(negative_pair.shape[0],1).cuda()
        # print(self.discriminator(positive_pair).shape)
        real_loss = self.loss_fn(self.discriminator(positive_pair), real)
        fake_loss = self.loss_fn(self.discriminator(negative_pair), fake)
        discrimination_loss = (real_loss + fake_loss) / 2

        score = (s_pp - s_np)#.squeeze() y d l+1 3662

        perm = topk(score, self.ratio, batch) #2979

        #print(batch.shape,perm.shape,batch[perm].shape)
        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1) #对应公式12 #2979x128

        batch = batch[perm]  #2979
        filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0)) # filter_edge_index: 2x4654
        print("x.shape--batch.shape",discrimination_loss)
        return x, filter_edge_index, filter_edge_attr, batch, perm, discrimination_loss