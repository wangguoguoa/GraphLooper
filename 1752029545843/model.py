import torch
from torch_geometric.nn import GCNConv, SAGEConv,GATConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool,GCN,HGPSLPool
import numpy as np
import torch.nn as nn

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from poolings.CGIPool import CGIPool


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sigmoid = nn.Sigmoid()

        self.conv1 = GCNConv(self.num_features, self.nhid,)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        # self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

        # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, 1)

    def forward(self, data):
        x, edge_index,edge_attr,pos,batch = data.x, data.edge_index, data.edge_attr,data.pos, data.batch

        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = self.sigmoid(self.lin2(x))

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.sigmoid(self.lin3(x))

        return x
    
class GrapConvNet(torch.nn.Module):
    def __init__(self, args):
        super(GrapConvNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.bn = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()

        self.graph1 = GCNConv(self.num_features, self.nhid,)
        self.graphpool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.graph2 = GCNConv(self.nhid, self.nhid)
        self.graphpool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.graph3 = GCNConv(self.nhid, self.nhid)
        self.graphpool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2, bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        c_in = 384
        # self.lin1 = torch.nn.Linear(c_in, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        # self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.lin1 = torch.nn.Linear(c_in, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, 1)
    
    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, data):
        x, edge_index,edge_attr,pos,strata_data,batch = data.x, data.edge_index, data.edge_attr,data.pos,data.strata_data,data.batch
        
        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)
        
        x = F.relu(self.graph1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.graphpool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.graph2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.graphpool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.graph3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _ = self.graphpool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        gc_outputs = x1 + x2 + x3

        out1 = self.conv1(strata_data)
        out1 = F.relu(out1)
        out1 = self.pool1(out1)
        out1 = F.dropout(out1)
        out1 = self.conv2(out1)
        out1 = F.relu(out1)
        out1 = self.pool2(out1)
        out1 = F.dropout(out1)
        out1 = self.conv3(out1)
        out1 = F.relu(out1)
        out1 = self.pool3(out1)
        out1 = F.dropout(out1)
        conv_outputs = self.bn(out1)
        conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)

        x = torch.tensor(np.concatenate((gc_outputs.cpu().detach().numpy(),conv_outputs.cpu().detach().numpy()),axis=-1),device=x.device)



        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = self.sigmoid(self.lin3(x))

        x = self.sigmoid(self.lin2(x))

        return x
    

class ConvNet(torch.nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.bn = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2, bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        c_in = 128

        self.lin1 = torch.nn.Linear(c_in, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, 1)
    
    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, data):
        x, edge_index,edge_attr,pos,strata_data,batch = data.x, data.edge_index, data.edge_attr,data.pos,data.strata_data,data.batch

        x = x.reshape((1,1,x.shape[0],x.shape[1]))

        x = torch.tensor(np.concatenate((strata_data.cpu(), x.cpu()), axis=1),device=x.device)
        

        out1 = self.conv1(x)
        out1 = F.relu(out1)
        out1 = self.pool1(out1)
        out1 = F.dropout(out1)
        out1 = self.conv2(out1)
        out1 = F.relu(out1)
        out1 = self.pool2(out1)
        out1 = F.dropout(out1)
        out1 = self.conv3(out1)
        out1 = F.relu(out1)
        out1 = self.pool3(out1)
        out1 = F.dropout(out1)
        conv_outputs = self.bn(out1)
        conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)


        x = F.relu(self.lin1(conv_outputs))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = self.sigmoid(self.lin3(x))

        x = self.sigmoid(self.lin2(x))

        return x



class SSPool(torch.nn.Module):
    def __init__(self,  args):
        super(SSPool, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sigmoid = nn.Sigmoid()

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.pool1 = CGIPool(self.nhid, ratio=self.pooling_ratio)
        self.pool2 = CGIPool(self.nhid, ratio=self.pooling_ratio)
        self.pool3 = CGIPool(self.nhid, ratio=self.pooling_ratio)

        # self.linear1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.linear2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        # self.linear3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.linear1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        # self.linear2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.linear2 = torch.nn.Linear(self.nhid, 1)

        self.dis_loss1, self.dis_loss2, self.dis_loss3 = None, None, None

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        x, edge_index,edge_attr,pos,strata_data,batch = data.x, data.edge_index, data.edge_attr,data.pos,data.strata_data,data.batch
        # print(x.device,edge_index.device,pos.device,batch.device)

        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)
        # print(x.device)

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, loss = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss1 = loss

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, loss = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss2 = loss

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, loss = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss3 = loss

        x = x1 + x2 + x3

        # x = F.relu(self.linear1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.linear2(x))
        # x = F.log_softmax(self.linear3(x), dim=1)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = self.sigmoid(self.lin3(x))

        x = self.sigmoid(self.linear2(x))

        return x

    def compute_disentangle_loss(self):
        return (self.dis_loss1 + self.dis_loss2 + self.dis_loss3) / 3


class GATNet(torch.nn.Module):
    def __init__(self,  args):
        super(GATNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GATConv(self.num_features, self.nhid)
        self.conv2 = GATConv(self.nhid, self.nhid)
        self.conv3 = GATConv(self.nhid, self.nhid)

        self.pool1 = CGIPool(self.nhid, ratio=self.pooling_ratio)
        self.pool2 = CGIPool(self.nhid, ratio=self.pooling_ratio)
        self.pool3 = CGIPool(self.nhid, ratio=self.pooling_ratio)

        self.linear1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.linear2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.linear3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.dis_loss1, self.dis_loss2, self.dis_loss3 = None, None, None

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        # print(x.device,edge_index.device,pos.device,batch.device)

        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)
        # print(x.device)

        x = F.relu(self.conv1(x, edge_index))
        # print(x.device)
        x, edge_index, _, batch, _, loss = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss1 = loss

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, loss = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss2 = loss

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, loss = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss3 = loss

        x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=1)

        return x

    def compute_disentangle_loss(self):
        return (self.dis_loss1 + self.dis_loss2 + self.dis_loss3) / 3


class GraphNet(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GATConv(self.num_features, self.nhid,)
        self.conv2 = GATConv(self.nhid, self.nhid)
        self.conv3 = GATConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index,pos,batch = data.x, data.edge_index,data.pos, data.batch
        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)


        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x



class HMTP(torch.nn.Module):
    def __init__(self, args):
        super(HMTP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        self.hidden_representations = []

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        # self.pool1=  TopKPooling(self.nhid, self.pooling_ratio)
        # self.pool2 = TopKPooling(self.nhid, self.pooling_ratio)
        #self.pool1 = SAGPooling
        self.sigmoid = nn.Sigmoid()
        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

    def forward(self, data,k):
        x, edge_index,edge_attr,pos,strata_data,batch = data.x, data.edge_index, data.edge_attr,data.pos,data.strata_data,data.batch
        # print(type(edge_index))
        x = torch.tensor(np.concatenate((x.cpu(), pos.cpu()), axis=1),device=x.device)
        

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # print(x.shape, batch.shape)
        # x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr,batch)
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, k, batch)
        # print(x.shape, batch.shape)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # x, edge_index, edge_attr, batch  = self.pool2(x, edge_index, edge_attr,batch)
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, k, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        # print(x.shape)
        self.hidden_representations.append(x)

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.log_softmax(self.lin3(x), dim=-1)


        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training) 
        output = self.sigmoid(self.lin3(x))


        return output
