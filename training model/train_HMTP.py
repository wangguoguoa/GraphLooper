#!/usr/bin/python

import os
import sys
import time
import argparse
import random
import torch
import numpy as np
import os.path as osp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
from torch_geometric.data import Data, DataLoader
# custom functions defined by user
from model import *
from loss import *
import scipy.sparse as sp
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, f1_score,recall_score,precision_recall_curve,auc
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt

from DQN import *
from QLearning import *
from env import GNN_env


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" GraphLopper train model for chromatin loops")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")

    parser.add_argument("-s", dest="seed", type=int, default=666,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=128,
                        help="Number of sequences sent to the network in one step.")
    # parser.add_argument("-lr", dest="learning_rate", type=float, default=0.0005,
    #                     help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-m", dest="momentum", type=float, default=0.9,
                        help="Momentum for the SGD optimizer.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=30,
                        help="Number of training steps.")
    # parser.add_argument("-w", dest="weight_decay", type=float, default=0.0001,
    #                     help="Regularisation parameter for L2-loss.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.001,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("-p", dest="power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("-r", dest="restore", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping')
    parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')

    return parser.parse_args()

def setup(rank, world_size):

    ###initialize the process group
    dist.init_process_group("ncll", rank=rank, world_size=world_size)

def main():
    """Create the model and start the training."""
    args = get_args()
    args.num_classes = 2
    args.num_features = 20
    torch.manual_seed()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        # print(torch.cuda.device_count())
        # print(torch.cuda.current_device())
    else:
        args.device = torch.device("cpu")
        torch.manual_seed(args.seed)
    def test(model, loader,k):
        model.eval()
        loss = 0.
        label_p_all, label_t_all = [], []
        for data in loader:
            data = data.to(args.device)
            with torch.no_grad():
                pred = model(data,k)
            label_p_all.append(pred.view(-1).data.cpu().numpy()[0])
            label_t_all.append(data.y.view(-1).data.cpu().numpy()[0])
        f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        precision, recall, _ = precision_recall_curve(label_t_all,[int(x > 0.5) for x in label_p_all])
        prauc = auc(recall, precision)
        return f1,prauc

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     args.device = torch.device("cuda")
    #     print('cuda is available')
    # else:
    #     args.device = torch.device("cpu")
    #     torch.manual_seed(args.seed)



    # #######在全基因组上######
    files = os.listdir(args.data_dir)
    chromnames = []
    for file in files:
        ccname = file.split('_')[0]
        if ccname not in chromnames:
            chromnames.append(ccname)

    f = open(osp.join(args.checkpoint, 'record.txt'), 'w')
    for chromname in chromnames:
        strata_te, pos_encoding_te, epis_te, label_te = [], [], [], []
        strata, pos_encoding, epis, label = [], [], [], []
        for filename in files:
            if filename.split('_')[0] == chromname:
                Data_temp_te = np.load(osp.join(args.data_dir, filename),allow_pickle=True)
                strata_te.extend(Data_temp_te['strata'])
                pos_encoding_te.extend(Data_temp_te['pos_encoding'])
                epis_te.extend(Data_temp_te['epis'])
                label_te.extend(Data_temp_te['label'])

            else:
                Data_temp = np.load(osp.join(args.data_dir, filename),allow_pickle=True)
                strata.extend(Data_temp['strata'])
                pos_encoding.extend(Data_temp['pos_encoding'])
                epis.extend(Data_temp['epis'])
                label.extend(Data_temp['label'])

        ###当前染色体数据
        strata_te = np.array(strata_te)
        pos_encoding_te = np.array(pos_encoding_te)
        epis_te = np.array(epis_te)
        label_te = np.array(label_te)
        label_te = label_te.reshape((label_te.shape[0], 1))

        dataset_te = []
        for i in range(len(strata_te)):
            edge_index_temp = sp.coo_matrix(strata_te[i])
            indice_te = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
            indice_te = torch.LongTensor(indice_te)  # 我们真正需要的coo形式

            value_te = edge_index_temp.data.astype(float)  # 边上对应权重值weight
            value_te = torch.FloatTensor(value_te)
            # value_te = value_te.astype(float)
            epis_feature_temp_te = torch.tensor(epis_te[i], dtype=torch.float)
            pos_encoding_temp_te = torch.tensor(pos_encoding_te[i], dtype=torch.float)
            label_temp_te = torch.tensor(label_te[i], dtype=torch.int64)
            strata_temp_te = torch.tensor(strata_te[i].astype(float), dtype=torch.float)
            # strata_temp_te = strata_te[i].astype(float)
            strata_temp_te = strata_temp_te.reshape((1,1,strata_temp_te.shape[0],strata_temp_te.shape[1]))

            data_temp_te = Data(x=epis_feature_temp_te, edge_index=indice_te, y=label_temp_te,
                                edge_attr=value_te, pos=pos_encoding_temp_te,strata_data=strata_temp_te)
            dataset_te.append(data_temp_te)
        print(len(dataset_te))

        #####剩余染色体数据
        strata = np.array(strata)
        pos_encoding = np.array(pos_encoding)
        epis = np.array(epis)
        label = np.array(label)
        label = label.reshape((label.shape[0], 1))

        dataset = []
        for i in range(len(strata)):
            edge_index_temp = sp.coo_matrix(strata[i])
            indice = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
            indice = torch.LongTensor(indice)  # 我们真正需要的coo形式

            value = edge_index_temp.data.astype(float) # 边上对应权重值weight

            value = torch.FloatTensor(value)
            # value = value.astype(float)
            epis_feature_temp = torch.tensor(epis[i], dtype=torch.float)
            pos_encoding_temp = torch.tensor(pos_encoding[i], dtype=torch.float)
            label_temp = torch.tensor(label[i], dtype=torch.int64)
            # strata_temp = torch.tensor(strata[i], dtype=torch.float)
            strata_temp = torch.tensor(strata[i].astype(float), dtype=torch.float)
            # strata_temp = strata[i].astype(float)
            strata_temp = strata_temp.reshape((1,1,strata_temp.shape[0],strata_temp.shape[1]))

            data_temp = Data(x=epis_feature_temp, edge_index=indice, y=label_temp,
                             edge_attr=value, pos=pos_encoding_temp,strata_data=strata_temp)
            dataset.append(data_temp)
        print(len(dataset))

        ratio = 0.8
        num_training = int(len(dataset) * ratio)
        num_val = len(dataset) - num_training
        training_set, validation_set = random_split(dataset, [num_training, num_val])
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset_te, batch_size=1, shuffle=False)

        # model = HMTP(args).to(args.device)
        # model = HMTP(args)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        min_loss = 1e10
        patience = 0
        f1_best, prauc_best = 0, 0
        k_step_value = 0.1
        env = GNN_env(action_value=k_step_value,
                  subgraph_num=args.num_features, initial_k=args.pooling_ratio)
        N_STATES=1#当前状态，只有一个状态
        dqn = DQN(env.n_actions,N_STATES,env.action_space)
        # RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=0.001)
        
        
        # if there exists multiple GPUs, using DataParallel
        model = HMTP(args).to(args.device)
        # model = Net()
        if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        for _ in range(5):
            k_record = [0.5]
            if args.restore:
                print("Resume it from {}.".format(args.restore_from))
                checkpoint = torch.load(args.restore)
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict, strict=False)
            model.to(args.device)
            for epoch in range(args.max_epoch):
                model.train()
                loss_train = 0.0
                sum = 0
                for i, data in enumerate(train_loader):
                    data = data.to(args.device)
                    optimizer.zero_grad()
                    out = model(data,args.pooling_ratio)
                    if np.isnan(out.float().cpu().detach().numpy()).any():
                        continue
                    criterion = OhemLoss()
                    loss = criterion(out.float(), data.y.float())
                    print("Training loss:{}".format(loss.item()))
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()
                    sum = sum +1  
                    # break
                val_f1, val_prauc = test(model, val_loader, args.pooling_ratio)
                loss_epoch = loss_train / sum
                limited_epoch = -6
                delta_k = 0.1
                k=args.pooling_ratio
                if epoch >=50:
                    if not isTerminal(k_record, limited_epochs=limited_epoch, delta_k=delta_k):
                        # k, reward = run_QL(env, RL, test, train_loader, loss_train)
                        k=run_dqn(env,epoch,dqn,model,test, train_loader, loss_train)
                        k=np.round(k,1)
                        if k==0 or k==1:
                            continue
                        k_record.append(k)
                        print("RL RUNING")
                        print(k)
                    else:
                        print("RL ENDING")
                        endingRLEpoch = epoch
                        break
                else:
                    k_record.append(k)
                print("Epoch：{},f1:{}\tprauc:{}\tloss_epoch:{}".format(epoch+1, val_f1, val_prauc,loss_epoch))

                if  f1_best < val_f1 :
                    f1_best = val_f1
                    prauc_best = val_prauc
                    checkpoint_file = osp.join(args.checkpoint, '{}_model_best.pth'.format(chromname))
                    torch.save(model.state_dict(), checkpoint_file)
                    # torch.save({
                    # 'model_state_dict': state_dict
                    # }, checkpoint_file)
                    patience = 0
                else:
                    patience += 1
                if patience > args.patience:
                    break
        model = HMTP(args).to(args.device)
        model.load_state_dict(torch.load(checkpoint_file))
        f1, prauc = test(model, test_loader,args.pooling_ratio)
        f.write("The test result on {} is \n prauc: {:.3f}\tf1: {:.3f}\n".format(chromname,f1, prauc))
    f.close()


if __name__ == "__main__":
    main()
