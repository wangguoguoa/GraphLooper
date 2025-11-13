#!/usr/bin/env python
import argparse
import gc
import pathlib
import os
import numpy as np
import torch
from scoreUtils import *
from utils import *
from model import *

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description=" score every possible chromatin loop")

    parser.add_argument("-p", dest="path", type=str, default=None,
                        help="Path to a .cool URI string or a .hic file.")
    parser.add_argument("--balance",dest="balance", default= False,
                        help="Whether or not using the ICE/KR-balanced matrix.")
    parser.add_argument("-a", dest="bigwig",type=str, default=None,
                        help="Path to the chromatin accessibility data which is a bigwig file ")
    parser.add_argument("-o", dest="output", default='./scores/', help="Folder path to store results.")
    parser.add_argument("-m", dest="model_path", default=None, help="Path to a trained mode.")
    parser.add_argument('-l', '--lower', type=int, default=2,
                   help='''Lower bound of distance between loci in bins (default 2).''')
    parser.add_argument('-u', '--upper', type=int, default=300,
                   help='''Upper bound of distance between loci in bins (default 300).''')
    parser.add_argument('-w', '--width', type=int, default=22,
                   help='''Number of bins added to center of window. 
                            default width=11 corresponds to 23*23 windows''')
    parser.add_argument('-r', '--resolution',
                   help='Resolution in bp, default 10000',
                   type=int, default=10000)
    parser.add_argument('--minimum-prob', type=float, default=0.5,
                   help='''Only output pixels with probability score greater than this value (default 0.5)''')
    
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

def main():
    args = get_args()
    args.num_classes = 2
    args.num_features = 20
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            args.device = torch.device("cuda:" + args.gpu)
        else:
            args.device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        args.device = torch.device("cpu")
    np.seterr(divide='ignore', invalid='ignore')

    # pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    # state_dict = checkpoint["state_dict"]
    model = HMTP(args).to(args.device)
    # deepcnn.load_state_dict(state_dict)
    model.load_state_dict(checkpoint)
    # model.to(torch.device("cuda:0"))

    # more robust to check if a file is .hic
    hic_info = read_hic_header(args.path)
    if hic_info is None:
        hic = False
    else:
        hic = True

    if not hic:
        import cooler
        Lib = cooler.Cooler(args.path)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(args.path, args.resolution)

    pre = find_chrom_pre(chromosomes)
    tmp = os.path.split(args.model_path)[1]  # support full path
    # ccname is consistent with chromosome labels in .hic / .cool
    ccname = pre + tmp.split('_model')[0].lstrip('chr')
    cikada = 'chr' + ccname.lstrip('chr')  # cikada always has prefix "chr"

    if not hic:
        M = tocsr(Lib.matrix(balance=False, sparse=True).fetch(ccname))
        X = Chromosome(M, model=model,bigwig=args.bigwig, raw_M=M,
                                  cname=cikada, lower=args.lower,
                                  upper=args.upper, res=args.resolution,
                                  width=args.width)
        

    else:
        if args.balance:
            X = Chromosome(csr_contact_matrix('KR', args.path, ccname, ccname, 'BP', args.resolution),
                                      model=model,bigwig=args.bigwig,
                                      cname=cikada, lower=args.lower,
                                      upper=args.upper, res=args.resolution,
                                      width=args.width,k=args.pooling_ratio)
        else:
            X = Chromosome(csr_contact_matrix('NONE', args.path, ccname, ccname, 'BP', args.resolution),
                                      model=model,
                                      cname=cikada, lower=args.lower,
                                      upper=args.upper, res=args.resolution,
                                      width=args.width)
    result, R = X.score(thre=args.minimum_prob)
    X.writeBed(args.output, result, R)

if __name__ == "__main__":
    main()