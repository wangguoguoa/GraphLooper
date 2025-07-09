#!/usr/bin/env python
import pathlib
import hicstraw
import argparse
import numpy as np
from dataUtils import *
from utils import *
import numpy as np
import pandas as pd

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="generate positive and negative samples ")

    parser.add_argument("--balance", dest="balance", default=False,
                        help="Whether or not using the ICE/KR-balanced matrix.")
    parser.add_argument('-l', '--lower', type=int, default=2,
                        help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=300,
                        help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=22,
                        help='''Number of bins added to center of window. 
                                default width=22 corresponds to 45*45 windows''')
    parser.add_argument('-r', '--resolution',
                        help='Resolution in bp, default 10000',type=int, default=10000)
    parser.add_argument('-genome', '--rg',
                        help='The genome reference ', default='hg19')


    return parser.parse_args()


def load_chrom_sizes(reference_genome):
    """
    Load chromosome sizes for a reference genome
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    rg_path = f'{my_path}/{reference_genome}.chrom.sizes'
    f = open(rg_path)
    lengths = {}
    for line in f:
        [ch, l] = line.strip().split()
        lengths[ch] = int(l)
    return lengths



def load_epigenetic_data(cell_line, chromname, epi_names, processed_path, verbose=1):

    res = 10000
    epi_data = None
    for i, k in enumerate(epi_names):
        src = '{0}/{1}/Epi/{2}/{3}_{4}bp_{5}.npy'.format(processed_path,cell_line, chromname, chromname, res, k)
        s = np.load(src)
        s = np.log(s + 1)
        s[s > 4] = 4
        if verbose:
            print(' Loading epigenomics...', chromname, k, len(s))
        if i == 0:
            epi_data = np.zeros((len(s), len(epi_names)))
        epi_data[:, i] = s
    return epi_data

def positional_encoding(length, position_dim=8):
    assert position_dim % 2 == 0
    position = np.zeros((length, position_dim))
    for i in range(position_dim // 2):
        position[:, 2 * i] = np.sin([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
        position[:, 2 * i + 1] = np.cos([pp / (10000 ** (2 * i / position_dim)) for pp in range(length)])
    return position
            
            
def generategraph(Matrix, coords, epis,width=22, lower=1,pos_enc_dim=8,positive=True, stop=5000):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """
    tp_strata =[]
    tp_pos_encoding=[]
    tp_epis = []
    negcount = 0 
    for c in coords:
        x, y = c[0], c[1]
        # x, y = 5101, 5108
        if y-x < lower:
            pass
        else:
            window = Matrix[x-width:x+width+1,
                            y-width:y+width+1].toarray()
            if window.size!= (2*width+1)*(2*width+1):
                continue
            if np.count_nonzero(window) < window.size*.1:
                pass
            else:
                center = window[width, width]
                ls = window.shape[0]
                p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                if positive and p2LL < 0.1:
                    pass
                else:
                    pos_encoding = positional_encoding((2*width+1), pos_enc_dim)
                    epi_x = epis[x-width:x+width+1]
                    epi_x = (epi_x - np.min(epi_x)) / (np.max(epi_x) - np.min(epi_x))
                    epi_y = epis[y-width:y+width+1]
                    if epi_y.shape[0] != 2*width+1:
                        print(x,y)
                        m = np.zeros((2*width+1-epi_y.shape[0],6)) 
                        epi_y = np.concatenate((epi_y,m),axis=0)

                    epi_y = (epi_y - np.min(epi_y)) / (np.max(epi_y) - np.min(epi_y))
                    epi = np.concatenate((epi_x,epi_y),axis=1)

                    window = (window - np.min(window)) / (np.max(window) - np.min(window))
                    tp_strata.append(window)
                    tp_pos_encoding.append(pos_encoding)
                    tp_epis.append(epi)
                    if not positive:
                        negcount += 1
                    if negcount >= stop:
                        break
    return tp_strata,tp_pos_encoding,tp_epis


if __name__ == "__main__":
    args = get_args()
    cell_line = 'K562_SKJ'
    raw_path = '/data/wangsiguo/graphloops/source_data'
    processed_path = '/data/wangsiguo/graphloops/processed_data_all'
    bedpe = '/data/wangsiguo/graphloops/source_data/training-sets/k562.encode.ctcf-chiapet.hg19.bedpe'
    # epi_names = ['DNase', 'H3K9ac', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3','CTCF']
    epi_names = ['DNase',  'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'CTCF']
    np.seterr(divide='ignore', invalid='ignore')
    # more robust to check if a file is .hic
    hic_info = read_hic_header(raw_path+'/{}/{}.cool'.format(cell_line,cell_line))
    if hic_info is None:
        hic = False
    else:
        hic = True
    if not hic:
        import cooler
        Lib = cooler.Cooler(raw_path+'/{}/{}.cool'.format(cell_line,cell_line))
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(args.path, args.resolution)
    ######   load annotating data
    coords = parsebed(bedpe, lower=2,res=args.resolution)  # 取标记的每对loop的start位置，并在每条染色体上进行排序，生成一个字典包含23条染色体上的所有正样本的两个start位置（除以了10000）
    kde, lower, long_start, long_end = learn_distri_kde(coords)

    
    #####  process graph 
    positive_strata = {}
    negative_strata ={}
    positive_labels = {}
    negative_labels = {}
    for key in chromosomes:
        if key == 'X':
            if key.startswith('chr'):
                chromname = key
            else:
                chromname = 'chr' + key
            print('collecting from {}'.format(chromname))
            if not hic:
                X = Lib.matrix(balance=False,
                            sparse=True).fetch(key).tocsr()
            else:
                if args.balance:
                    X = csr_contact_matrix(
                        'KR', args.path, key, key, 'BP', args.resolution)
                else:
                    X = csr_contact_matrix(
                        'NONE', args.path, key, key, 'BP', args.resolution)
            clist = coords[chromname]  #  save per  annotating data

            epi_features = load_epigenetic_data(cell_line, chromname, epi_names, processed_path)
            #####generate positive samples
            pos_strata,pos_posen_encoding,pos_epis=generategraph(X, clist, epi_features, width=22, pos_enc_dim=8)
            strata = np.array(pos_strata)
            pos_encoding = np.array(pos_posen_encoding)
            epis = np.array(pos_epis)
            positive_strata[chromname]=[pos_strata,pos_posen_encoding,pos_epis]
            print(f'the size of {chromname} positive sample is {len(pos_strata)} {len(pos_posen_encoding)} {len(pos_epis)} ')
            positive_labels[chromname] = np.ones(len(pos_strata))
            np.savez(processed_path +'/{}/sample/'.format(cell_line) + '%s_positive.npz' % chromname,
                    strata=positive_strata[chromname][0],pos_encoding=positive_strata[chromname][1],
                    epis=positive_strata[chromname][2],label=positive_labels[chromname])

            ###########generate negative samples
            neg_coords = negative_generating(X, kde, clist, lower, long_start, long_end)
            stop = len(clist)
            neg_strata, neg_posen_encoding, neg_epis = generategraph(X, neg_coords, epi_features, width=22,
                                                                pos_enc_dim=8,positive=False,stop=stop)
            
            # print(neg_strata.shape)
            neg_strata = np.array(neg_strata,dtype=object)
            neg_posen_encoding = np.array(neg_posen_encoding)
            neg_epis = np.array(neg_epis)
            negative_strata[chromname] = [neg_strata, neg_posen_encoding, neg_epis]
            print(f'the size of {chromname} negative sample is {len(neg_strata)} {len(neg_posen_encoding)} {len(neg_epis)} ')
            negative_labels[chromname] = np.zeros(len(neg_strata))
            np.savez(processed_path + '/{}/sample/'.format(cell_line) + '%s_negative.npz' % chromname,
                    strata=negative_strata[chromname][0], pos_encoding=negative_strata[chromname][1],
                    epis=negative_strata[chromname][2],label=negative_labels[chromname])



