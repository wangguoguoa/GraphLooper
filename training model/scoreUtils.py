from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy import stats
from scipy.ndimage import gaussian_filter
from utils import *
import torch
from torch_geometric.data import Data, DataLoader
import scipy.sparse as sp
#from statsmodels.sandbox.stats.multicomp import multipletests

class Chromosome():
    def __init__(self, M, model,bigwig, raw_M=None, weights=None,
        lower=6, upper=300, cname='chrm', res=10000, width=5,k=0.5):
        
        lower = max(lower, width+1)
        upper = min(upper, M.shape[0]-2*width)
        # calculate expected values
        if weights is None:
            self.exp_arr = calculate_expected(M, upper+2*width, raw=True)
            if M is raw_M:
                self.background = self.exp_arr
            else:
                self.background = calculate_expected(raw_M, upper+2*width, raw=True)
        else:
            self.exp_arr = calculate_expected(M, upper+2*width, raw=False)
            self.background = self.exp_arr
        
        self.raw_M = raw_M
        self.weights = weights
        self.k = k

        # lower down the memory usage
        R, C = M.nonzero()
        validmask = np.isfinite(M.data) & (C-R > (-2*width)) & (C-R < (upper+2*width))
        R, C, data = R[validmask], C[validmask], M.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=M.shape)
        self.get_candidate(lower, upper)
        self.chromname = cname
        self.r = res
        self.w = width
        self.model = model
        self.bigwig = bigwig
    
    def get_candidate(self, lower, upper):

        x_arr = np.array([], dtype=int)
        y_arr = np.array([], dtype=int)
        p_arr = np.array([], dtype=float)
        idx = np.arange(self.raw_M.shape[0])
        for i in range(lower, upper+1):
            diag = self.raw_M.diagonal(i)
            e = self.background[i]
            if (diag.size > 0) and (e > 0):
                xi = idx[:-i]
                yi = idx[i:]
                if self.weights is None:
                    exp = np.ones(diag.size, dtype=float) * e
                else:
                    b1 = self.weights[:-i]
                    b2 = self.weights[i:]
                    exp = np.ones(diag.size, dtype=float) * e / (b1 * b2)
                
                Poiss = stats.poisson(exp)
                pvalues = Poiss.sf(diag)
                mask = (diag > 0) & np.isfinite(pvalues)
                x_arr = np.r_[x_arr, xi[mask]]
                y_arr = np.r_[y_arr, yi[mask]]
                p_arr = np.r_[p_arr, pvalues[mask]]
        
        #qvalues = multipletests(p_arr, method = 'fdr_bh')[1]
        mask = p_arr < 0.01
        self.ridx, self.cidx = x_arr[mask], y_arr[mask]
    
    def load_epigenetic_data(self,bigwig,verbose=1):
    
        cell_line = 'K562'
        chromname = self.chromname
        epi_names = ['DNase',  'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'CTCF']
        res = 10000
        epi_data = None
        for i, k in enumerate(epi_names):
            src = '{0}/{1}/{2}_{3}bp_{4}.npy'.format(bigwig,chromname, chromname, res, k)
            s = np.load(src)
            s = np.log(s + 1)
            s[s > 4] = 4
            # if verbose:
            #     print(' Loading epigenomics...', chromname, k, len(s))
            if i == 0:
                epi_data = np.zeros((len(s), len(epi_names)))
            epi_data[:, i] = s
        return epi_data

    def positional_encoding(self,length, position_dim=8):
        assert position_dim % 2 == 0
        position = np.zeros((length, position_dim))
        for i in range(position_dim // 2):
            position[:, 2 * i] = np.sin([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
            position[:, 2 * i + 1] = np.cos([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
        return position
    
    def generategraph(self,windows,coords, epis,width=22, lower=1,pos_enc_dim=8):
        tp_strata =[]
        tp_pos_encoding=[]
        tp_epis = []
        negcount = 0 
        for c in coords:
            x, y = c[0], c[1]
            pos_encoding = self.positional_encoding((2*width+1), pos_enc_dim)
            epi_x = epis[x-width:x+width+1]
            epi_x = (epi_x - np.min(epi_x)) / (np.max(epi_x) - np.min(epi_x))
            epi_y = epis[y-width:y+width+1]
            if epi_y.shape[0] != 2*width+1:
                m = np.zeros((2*width+1-epi_y.shape[0],6)) 
                epi_y = np.concatenate((epi_y,m),axis=0)
            epi_y = (epi_y - np.min(epi_y)) / (np.max(epi_y) - np.min(epi_y))
            epi = np.concatenate((epi_x,epi_y),axis=1)
            tp_epis.append(epi)
            tp_pos_encoding.append(pos_encoding)
        for w in windows:
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            tp_strata.append(w)
        return tp_strata,tp_pos_encoding,tp_epis
    


    def generatedata(self,strata,pos_encoding,epis):
        dataset = []   
        for i in range(len(strata)):
            edge_index_temp = sp.coo_matrix(strata[i])
            indice_te = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
            indice_te = torch.LongTensor(indice_te)  # 我们真正需要的coo形式

            value_te = edge_index_temp.data  # 边上对应权重值weight
            value_te = torch.FloatTensor(value_te)
            epis_feature_temp_te = torch.tensor(epis[i], dtype=torch.float)
            pos_encoding_temp_te = torch.tensor(pos_encoding[i], dtype=torch.float)
            strata_temp_te = torch.tensor(strata[i], dtype=torch.float)
            strata_temp_te = strata_temp_te.reshape((1,1,strata_temp_te.shape[0],strata_temp_te.shape[1]))

            data_temp_te = Data(x=epis_feature_temp_te, edge_index=indice_te,
                                edge_attr=value_te, pos=pos_encoding_temp_te,strata_data=strata_temp_te)
            dataset.append(data_temp_te)
        return dataset
    def getwindow(self, coords):
        
        w = self.w
        coords = np.r_[coords]
        xi, yi = coords[:,0], coords[:,1]
        mask = (xi - w >= 0) & (yi + w + 1 <= self.M.shape[0])
        xi, yi = xi[mask], yi[mask]
        seed = np.arange(-w, w+1)
        delta = np.tile(seed, (seed.size, 1))
        xxx = xi.reshape((xi.size, 1, 1)) + delta.T
        yyy = yi.reshape((yi.size, 1, 1)) + delta
        v = np.array(self.M[xxx.ravel(), yyy.ravel()]).ravel()
        vvv = v.reshape((xi.size, seed.size, seed.size))
        windows, clist = distance_normalize(vvv, self.exp_arr, xi, yi, w)
        # print(np.array(windows).shape)
        epi_features = self.load_epigenetic_data(self.bigwig)
        tp_strata,tp_pos_encoding,tp_epis = self.generategraph(windows,clist, epi_features,width=22)
        data = self.generatedata(tp_strata,tp_pos_encoding,tp_epis)

        return data, clist
    
    def test(self, fts):
        fts = DataLoader(fts, batch_size=1, shuffle=False)
        preds = np.array([])
        for data in fts:
            # segment = fts[i*batch:(i+1)*batch]
            # segment = torch.from_numpy(segment)
            # segment = segment.float().to(torch.device("cuda:0"))
            with torch.no_grad():
                label_p = self.model(data,self.k)
            probas = label_p.view(-1).data.cpu().numpy()
            preds = np.concatenate((preds, probas))
            # print("The current number is {}".format((i+1)*batch))

        return preds

    def score(self, thre=0.5):

        print('scoring matrix {}'.format(self.chromname))
        print('number of candidates {}'.format(self.ridx.size))
        total_coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        print(len(total_coords))
        prob_pool = np.r_[[]]
        # to lower down the memory usage
        batch_size = 100000
        ri = np.r_[[]]
        ci = np.r_[[]]
        prob_pool = np.r_[[]]
        for t in range(0, len(total_coords), batch_size):
            print("The current number is {}".format((t+1)*batch_size))
            coords = total_coords[t:t+batch_size]
            fea, clist = self.getwindow(coords)
            if len(fea) > 1:
                p = self.test(fea)
                pfilter = p > thre
                clist = np.array(clist)
                ri = np.r_[ri, clist[:, 0][pfilter]]
                ci = np.r_[ci, clist[:, 1][pfilter]]
                prob_pool = np.r_[prob_pool, p[pfilter]]
        result = sparse.csr_matrix((prob_pool, (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M


    def writeBed(self, out, prob_csr, raw_csr):
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]
                output_bed.write('\t'.join(list(map(str, line)))+'\n')



# class Chromosome():
#     def __init__(self, coomatrix, model, ATAC,lower=1, upper=500, cname='chrm', res=10000, width=11 ):
#         # cLen = coomatrix.shape[0] # seems useless
#         R, C = coomatrix.nonzero()
#         validmask = np.isfinite(coomatrix.data) & (
#             C-R+1 > lower) & (C-R < upper)
#         R, C, data = R[validmask], C[validmask], coomatrix.data[validmask]
#         self.M = sparse.csr_matrix((data, (R, C)), shape=coomatrix.shape)
#         self.ridx, self.cidx = R, C
#         self.ATAC = ATAC
#         self.chromname = cname
#         self.r = res
#         self.w = width
#         self.model = model

#     def getwindow(self, coords):
#         """
#         Generate training set
#         :param Matrix: single chromosome dense array
#         :param coords: List of tuples containing coord bins
#         :param width: Distance added to center. width=5 makes 11x11 windows
#         :return: yields paired positive/negative samples for training
#         """
#         out_dir = '/home/sc3/wsg/deeploop/models/'
#         bw = pyBigWig.open(self.ATAC)
#         seq,clist,atac = [],[],[]
#         width = self.w
#         for i, c in enumerate(coords):
#             if (i+1) % 1000 == 0:
#                 print("The current iteration is {}".format(i+1))
#             # if i == 100:
#             #     break
#             x, y = c[0], c[1]
#             try:
#                 window = self.M[x-width:x+width+1,
#                                 y-width:y+width+1].toarray()
#             except:
#                 continue
#             if np.count_nonzero(window) < window.size*.2:
#                 pass
#             if np.isfinite(window).all() and window.shape == (2*width+1,2*width+1):
#                 try:
#                     window_x = np.array(bw.values(self.chromname, (x - width) * self.r, (x + width + 1) * self.r))
#                     window_x[np.isnan(window_x)] = 0
#                     window_x = window_x.reshape(2*width+1, self.r)
#                     window_x = [window_x.mean(axis=1)]
#                     window_y = np.array(bw.values(self.chromname, (y - width) * self.r, (y + width + 1) * self.r))
#                     window_y[np.isnan(window_y)] = 0
#                     window_y = window_y.reshape(2*width+1, self.r)
#                     window_y = [window_y.mean(axis=1)]
#                     window_atac = np.dot(np.transpose(window_x), window_y)
#                     seq.append(window)
#                     clist.append(c)
#                     atac.append(window_atac)
#                 except:
#                     continue
#         seq = np.array(seq)
#         atac = np.array(atac)
#         seq = seq.reshape((seq.shape[0], 1, seq.shape[1], seq.shape[2]))
#         atac = atac.reshape((atac.shape[0], 1, atac.shape[1], atac.shape[2]))
#         for i in range(len(seq)):
#             seq[i] = seq[i] / np.max(seq[i]+1)
#             atac[i] = np.log10(1 + atac[i] * 10)
#             atac[i] = atac[i] / np.max(atac[i]+1)
#         fts = np.concatenate((seq, atac), axis=1)
#         # np.savez(out_dir + '/{}_sample.npz'.format(self.chromname), data=fts, clist=clist)

#         return fts, clist

    # def test(self, fts):
    #     num_total = len(fts)
    #     batch = 20
    #     iteration = int(np.ceil(num_total/batch))
    #     preds = np.array([])
    #     for i in range(iteration):
    #         segment = fts[i*batch:(i+1)*batch]
    #         segment = torch.from_numpy(segment)
    #         # segment = segment.float().to(torch.device("cuda:0"))
    #         with torch.no_grad():
    #             label_p = self.model(segment,self.k)
    #         probas = label_p.view(-1).data.cpu().numpy()
    #         preds = np.concatenate((preds, probas))
    #         print("The current number is {}".format((i+1)*batch))

    #     return preds




    # def score(self, thre=0.5):
    #     print('scoring matrix {}'.format(self.chromname))
    #     print('num candidates {}'.format(self.M.data.size))
    #     coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
    #     fts, clist = self.getwindow(coords)
    #     p = self.test(fts)
    #     clist = np.r_[clist]
    #     pfilter = p > thre
    #     ri = clist[:, 0][pfilter]
    #     ci = clist[:, 1][pfilter]
    #     result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
    #     data = np.array(self.M[ri, ci]).ravel()
    #     self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

    #     return result, self.M

    # def writeBed(self, out, prob_csr, raw_csr):
    #     pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    #     with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
    #         r, c = prob_csr.nonzero()
    #         for i in range(r.size):
    #             line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
    #                     self.chromname, c[i]*self.r, (c[i]+1)*self.r,
    #                     prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]
    #             output_bed.write('\t'.join(list(map(str, line)))+'\n')
