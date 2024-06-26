import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import six
import sys
sys.modules['sklearn.externals.six'] = six
import random
from tqdm import tqdm

import networkx as nx
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from utils import identify_functional_groups, GetRingSystems


def dist_array(smiles = None, mols = None):
    if mols == None:
        l = len(smiles)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        l = len(mols)
    '''
    You can replace the Tanimoto distances of ECFPs with other molecular distance metrics!
    '''
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

    
def diversity_all(smiles = None, mols = None, dists = None, mode = "HamDiv", args = None, disable = False):
    if mode == "Richness":
        if smiles != None:
            return len(set(smiles))
        else:
            smiles = set()
            for mol in mols:
                smiles.add(Chem.MolToSmiles(mol))
            return len(smiles)
    elif mode == "FG":
        func_groups = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            smi = Chem.MolToSmiles(mols[i]) if smiles is None else smiles[i]
            grps = identify_functional_groups(smi)
            func_groups.update(grps)

        return(len(func_groups))

    elif mode == "RS":
        ring_sys = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            smi = Chem.MolToSmiles(mols[i]) if smiles is None else smiles[i]
            grps = GetRingSystems(smi)
            ring_sys.update(grps)

        return(len(ring_sys))

    elif mode == "BM":
        scaffolds = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            if mols is not None:
                scaf = MurckoScaffold.GetScaffoldForMol(mols[i])
            else:
                mol = Chem.MolFromSmiles(smiles[i])
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf)
            scaffolds.update([scaf_smi])

        return(len(scaffolds))


    if type(dists) is np.ndarray:
        l = len(dists)
    elif mols == None:
        l = len(smiles)
        assert l >= 2
        dists = dist_array(smiles)
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
        total = HamDiv(dists=dists)
        return total
    
    else:
        raise Error('Mode Undefined!')


def HamDiv(smiles = None, mols = None, dists=None, method="greedy_tsp"):
    l = dists.shape[0] if dists is not None else len(mols) if mols is not None else len(smiles)
    if l == 1:
        return 0
    dists = dist_array(smiles) if dists is None else dists
    
    remove = np.zeros(l)
    for i in range(l):
        for j in range(i + 1, l):
            if dists[i, j] == 0:
                remove[i] = 1
    remove = np.argwhere(remove == 1)
    dists = np.delete(dists, remove, axis = 0)
    dists = np.delete(dists, remove, axis = 1)
    
    G = nx.from_numpy_array(dists)
    
    if method == "exact_dp":
        tsp, total = solve_tsp_dynamic_programming(dists)
    elif method == "christofides":
        tsp = nx.approximation.christofides(G, weight='weight')
    elif method == "greedy_tsp":
        tsp = nx.approximation.greedy_tsp(G, weight='weight')
    elif method == "simulated_annealing_tsp":
        tsp = nx.approximation.simulated_annealing_tsp(G, init_cycle="greedy", weight='weight')
    elif method == "threshold_accepting_tsp":
        tsp = nx.approximation.threshold_accepting_tsp(G, init_cycle="greedy", weight='weight')
    elif method == "local_search":
        tsp, total = solve_tsp_local_search(dists, max_processing_time=300)
    else:
        Exception("Undefined method")
    
    if method not in ["exact_dp", "local_search"]:
        total = 0
        for i in range(1, len(tsp)):
            total += dists[tsp[i - 1], tsp[i]]
    
    return total