import os
import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest']


class EPIDataSetTrain(data.Dataset):
    def __init__(self, strata_tr,pos_encoding_tr,epis_tr, label_tr):
        super(EPIDataSetTrain, self).__init__()
        self.strata = strata_tr
        self.pos_encoding = pos_encoding_tr
        self.epis = epis_tr   
        self.label = label_tr

        assert len(self.strata) == len(self.label) and len(self.label) == len(self.pos_encoding)\
            and len(self.pos_encoding)== len(self.epis), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        strata_one = self.strata[index]
        pos_encoding_one = self.pos_encoding[index]
        epis_one = self.epis[index]
        label_one = self.label[index]

        return {"strata": strata_one, "pos_encoding": pos_encoding_one, 
                "epis": epis_one, "label": label_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, strata_te, pos_encoding_te, epis_te, label_te):
        super(EPIDataSetTest, self).__init__()
        self.strata = strata_te
        self.pos_encoding = pos_encoding_te
        self.epis = epis_te
        self.label = label_te
        assert len(self.strata) == len(self.label) and len(self.label) == len(self.pos_encoding)\
            and len(self.pos_encoding) == len(self.epis), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        strata_one = self.strata[index]
        pos_encoding_one = self.pos_encoding[index]
        epis_one = self.epis[index]
        label_one = self.label[index]

        return {"strata": strata_one, "pos_encoding": pos_encoding_one, 
                "epis": epis_one, "label": label_one}

def training_data_generator(strata, pos_encoding, epis, label, batch_size, n_epoches):
    pos_encoding = np.array([pos_encoding for _ in range(batch_size)])
    epis = np.array([epis for _ in range(batch_size)])

    # Initiating iteration:
    idx = list(range(len(strata)))
    np.random.seed(0)

    print('Start training:')

    for _epoch in range(n_epoches):
        print('Epoch:', _epoch + 1)
        np.random.shuffle(idx)

        for _batch in range(len(idx) // batch_size): ()

        print(' Batch:', _batch + 1)
        batch_idx = idx[_batch * batch_size: (_batch + 1) * batch_size]

        strata_batch = np.array(strata[_id] for _id in batch_idx)
        pos_encoding_batch = np.array(pos_encoding[_id] for _id in batch_idx)
        epis_batch = np.array(epis[_id] for _id in batch_idx)
        label_bath = np.array(label[_id] for _id in batch_idx)

        yield _epoch + 1, _batch + 1, (strata_batch, pos_encoding_batch, epis_batch), label_bath



