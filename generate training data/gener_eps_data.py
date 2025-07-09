#!/usr/bin/env python
import pathlib
import hicstraw
import argparse
import numpy as np
from dataUtils import *
from utils import *

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
    parser.add_argument('-w', '--width', type=int, default=11,
                        help='''Number of bins added to center of window. 
                                default width=11 corresponds to 23*23 windows''')
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



def load_bigWig_for_entire_genome(path, name, rg,resolution, epi_path=''):
    """
        Load bigwig file and save the signal as a 1-D numpy array

        Args:
            path (str): recommended: {cell_type}_{assay_type}_{reference_genome}.bigwig (e.g., mESC_CTCF_mm10.bigwig)
            name (str): the name for the epigenetic mark
            rg (str): reference genome
            resolution (int): resolution
            epi_path (str): the folders for storing data of all chromosomes

        No return value
        """
    if not os.path.exists(f'{epi_path}'):
        os.mkdir(f'{epi_path}')

    chromosome_sizes = load_chrom_sizes(rg)
    del chromosome_sizes['chrY']
    chroms = chromosome_sizes.keys()

    bw = pyBigWig.open(path)

    for ch in chroms:
        end_pos = chromosome_sizes[ch]
        nBins = end_pos // args.resolution
        end_pos = nBins * args.resolution  # remove the 'tail'

        vec = bw.stats(ch, 0, end_pos, exact=True, nBins=nBins)
        for i in range(len(vec)):
            if vec[i] is None:
                vec[i] = 0

        if not os.path.exists(f'{epi_path}/{ch}/'):
            os.mkdir(f'{epi_path}/{ch}/')
        np.save(f'{epi_path}/{ch}/{ch}_{resolution}bp_{name}.npy', vec)



def bigWig_bedGraph_to_vector(epi_names, raw_path, processed_path):
    for name in epi_names:
        if name == 'ATAC' or name == 'DNase':
            if os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_ATAC.bigWig'):
                file, _type = f'{raw_path}/{cell_line}/{cell_line}_ATAC.bigWig', 'bw'
            elif os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_DNase.bigWig'):
                file, _type = f'{raw_path}/{cell_line}/{cell_line}_DNase.bigWig', 'bw'
            else:
                raise ValueError(f'{cell_line} - {name}: data not found')
        else:
            if os.path.exists(f'{raw_path}/{cell_line}/{cell_line}_{name}.bigWig'):
                file, _type = f'{raw_path}/{cell_line}/{cell_line}_{name}.bigWig', 'bw'
            else:
                raise ValueError(f'{cell_line} - {name}: data not found')

        if _type == 'bw':
            load_bigWig_for_entire_genome(file, name,
                                          rg=args.rg, resolution=args.resolution, epi_path=f'{processed_path}/{cell_line}/Epi')
            print("{} is being processed".format(name))
            


if __name__ == "__main__":
    args = get_args()
    cell_line = 'GM12878'
    raw_path = '/data/wangsiguo/graphloops/source_data'
    processed_path = '/data/wangsiguo/graphloops/processed_data_all'
    # bedpe = '/data/wangsiguo/graphloops/data/training-sets/h1esc.4dn.ctcf-chiapet.hg38.bedpe'
    epi_names = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K27ac', 'H3K27me3','CTCF']
    # epi_names = ['CTCF']
    np.seterr(divide='ignore', invalid='ignore')

    
    ###   process epigenomic data
    bigWig_bedGraph_to_vector(epi_names, raw_path, processed_path)
    
