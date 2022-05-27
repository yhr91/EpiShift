import bioframe as bf
import pybedtools
import glob
import pandas as pd
import numpy as np

from torch import Tensor
from torch import nn
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

from multiprocessing import Pool
import tqdm

from sklearn.model_selection import train_test_split

def df_into_specified_bin_widths(
    df: pd.DataFrame, chromosome: str, bins: np.ndarray, *,
    labels = None, value_col = "value_") -> pd.DataFrame:
    """Redistribute signals from a data frame into specified bins.

    Parameters
    ----------
    df : pd.DataFrame
        File path to bedgraph file being redistributed; should have
    chromosome : str
        Chromosome identifier
    bins : np.ndarray (N, 2) of int
        Bin edges into which to redistribute signals, where rows indicate bins
        and columns identify the starting and ending genomic indices for the
        associated bin
    labels : Optional[Sequence[str]] (N,)
        Labels associated with bins (default = None); must be equal in length
        to the number of bins into which signals are being distributed
    value_col: Optional[str]
        Name of the column containing signal values (default = "value_")

    Returns
    -------
    pd.DataFrame
        Table with redistributed signals
    """
    chrom_col = np.array([chromosome for _ in range(len(bins))])
    bins_df = pd.DataFrame(
        data={
            "chrom": chrom_col,
            "start": bins[:, 0],
            "end": bins[:, 1]
        }
    )
    if labels is not None and len(labels) == len(bins_df):
        bins_df["label"] = labels
    overlap = bf.overlap(bins_df, df, how="left", return_overlap="True")
    return scale_overlap(overlap, value_col)


def scale_overlap(
    overlap: pd.DataFrame, value_col = "value_") -> pd.DataFrame:
    """Process overlap signals obtained from `bf.overlap`

    Notes
    -----
    Distributes signals so that they are proportional to the width of the
    bins in the original dataframe.

    Parameters
    ----------
    overlap : pd.DataFrame
        Original dataframe produced from `bf.overlap` containing `chrom`,
        `start`, `end`, `true_start`, `true_end`, and `value_` columns
    value_col: Optional[str]
        Name of the column containing signal values (default = "value_")

    Returns
    --------
    pd.DataFrame
        Processed signals, containing `chrom`, `start`, `end`, and
        `value_scaled` columns
    """
    new_scale = (overlap['True_end'] - overlap['True_start']).to_numpy()
    original_width = (overlap['end_'] - overlap['start_']).to_numpy()
    overlap["value_scaled"] = overlap[value_col] / original_width * new_scale
    overlap["value_scaled"] = overlap.groupby(
        ["chrom", "start", "end"]
    )["value_scaled"].transform("sum")
    processed_sigs = overlap.drop_duplicates(subset=['chrom', 'start', 'end'])
    if "label" in list(overlap):
        processed_sigs = \
            processed_sigs[["chrom", "start", "end", "value_scaled", "label"]]
    else:
        processed_sigs = \
            processed_sigs[["chrom", "start", "end", "value_scaled"]]
    return processed_sigs

def read_bigwig(fname, chr_=2):
    return bf.read_bigwig(fname, chrom='chr'+str(chr_))

def get_chrom_df(fname, chr_=2):
    
    bw = read_bigwig(fname, chr_)
    
    bin_size = 1000
    col1 = np.arange(0,len(bw),bin_size)[:-1]
    col2 = np.arange(0,len(bw),bin_size)[1:]
    bins = np.array([col1, col2]).T

    chrom_df = df_into_specified_bin_widths(bw, 'chr2', bins=bins)
    
    return chrom_df

def write_bigwig_torch(f):
    out = []
    BSS_name = f.split('BSS')[-1].split('.')[0]
    print(BSS_name)
    for chr_ in range(1,23):
        vals = get_chrom_df(f, chr_)['value_scaled'].values
        out.append(Tensor(vals.astype('float')))
    
    with open('torch_data/'+BSS_name+'.data', 'wb') as f:
        torch.save(out, f)
        
def Map(F, x, workers):
    """
    wrapper for map()
    Spawn workers for parallel processing
    
    """
    with Pool(workers) as pool:
        ret = list(tqdm.tqdm(pool.imap(F, x), total=len(x)))
    return ret

def mapper(f):
    write_bigwig_torch(f)
    return

bigwig_fnames = glob.glob('EpiMap/*bigWig')
Map(mapper, bigwig_fnames, workers=50)