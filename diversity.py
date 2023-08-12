import numpy as np
import time
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import six
import sys
sys.modules['sklearn.externals.six'] = six
import random
from tqdm import tqdm

def dist_array(smiles = None, mols = None):
    if mols == None:
        l = len(smiles)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        l = len(mols)
    sims = np.zeros((l, l))    
    fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(x, radius = 2, nBits = 2048) for x in mols]
    disable = (l <= 2000)
    for i in tqdm(range(l), disable = (l < 2000)):
        sims[i, i] = 1
        for j in range(i + 1, l):
            sims[i, j] = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            sims[j, i] = sims[i, j]
    dists = 1 - sims
    return dists
    
    
def diversity_all(smiles = None, mols = None, dists = None, mode = "IntDiv", args = None, disable = False):
    if type(dists) is np.ndarray:
        l = len(dists)
    elif mols == None:
        l = len(smiles)
        assert l >= 2
        dists = dist_array(smiles, mols)
    else:
        l = len(mols)
        assert l >= 2
        dists = dist_array(smiles, mols)
    
    if mode == "IntDiv":
        if l == 1:
            return 0
        return np.sum(dists) / l / (l - 1)
    elif mode == "SumDiv":
        if l == 1:
            return 0
        return np.sum(dists)/ (l - 1)
    elif mode == "Diam":
        if l == 1:
            return 0
        d_max = 0
        for i in range(l):
            for j in range(i + 1, l):
                if d_max < dists[i, j]:
                    d_max = dists[i, j]
        return d_max
    elif mode == "SumDiam":
        if l == 1:
            return 0
        sum_d_max = 0
        for i in range(l):
            d_max_i = 0
            for j in range(l):
                if j != i and d_max_i < dists[i, j]:
                    d_max_i = dists[i, j]
            sum_d_max += d_max_i
        return sum_d_max
    elif mode == "Bot":
        if l == 1:
            return 0
        d_min = 1
        for i in range(l):
            for j in range(i + 1, l):
                if d_min > dists[i, j]:
                    d_min = dists[i, j]
        return d_min
    elif mode == "SumBot":
        if l == 1:
            return 0
        sum_d_min = 0
        for i in range(l):
            d_min_i = 1
            for j in range(l):
                if j != i and d_min_i > dists[i, j]:
                    d_min_i = dists[i, j]
            sum_d_min += d_min_i
        return sum_d_min
    elif mode == "DPP":
        return np.linalg.det(1 - dists)
    elif mode == "Richness":
        if smiles != None:
            return len(set(smiles))
        else:
            smiles = set()
            for mol in mols:
                smiles.add(Chem.MolToSmiles(mol))
            return len(smiles)
        
    elif mode.split('-')[0] == 'NCircles':
        threshold = float(mode.split('-')[1])
        circs_sum = []
        for k in tqdm(range(1), disable = disable):
            circs = np.zeros(l)
            rs = np.arange(l)
            # random.shuffle(rs)
            for i in rs:
                circs_i = 1
                for j in range(l):
                    if j != i and circs[j] == 1 and dists[i, j] <= threshold:
                        circs_i = 0
                        break
                circs[i] = circs_i
            circs_sum.append(np.sum(circs))
        return np.max(np.array(circs_sum))
            
    elif mode == "HamDiv":
        if l == 1:
            return 0
        alpha = 0 if args == None else args
        total = 0
        remove = np.zeros(l)
        for i in range(l):
            d_i_min = 1
            for j in range(l):
                if j != i and remove[j] == 0 and d_i_min > dists[i, j]:
                    d_i_min = dists[i, j]
            if d_i_min <= alpha: # acceleration parameter
                remove[i] = 1
                total += 2 * d_i_min
        remove = np.argwhere(remove == 1)
        dists = np.delete(dists, remove, axis = 0)
        dists = np.delete(dists, remove, axis = 1)
        
        total, seq = greedy_TSP(dists, disable = disable)
        total = 0
        
        for i in range(1, len(seq)):
            total += dists[seq[i - 1], seq[i]]
        total += dists[seq[0], seq[-1]]
        
        return total
    else:
        raise Error('Mode Undefined!')


def greedy_TSP(dists, start_index = 0, disable = False):
    l = dists.shape[0]
    
    seq_result = [start_index]
    total = 0
    for k in tqdm(range(l - 1), disable = disable):
        distance_list = dists[start_index]
        min_dis = 2
        for i in range(l):
            if (min_dis > distance_list[i]) and (i not in seq_result):
                min_dis = distance_list[i]
                start_index = i
        total += min_dis
        seq_result += [start_index]
        
    total += dists[seq_result[0], seq_result[-1]]
    return total, seq_result