from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from collections import namedtuple

def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)

# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)
    
def identify_functional_groups(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        marked = set()
        #mark all heteroatoms in a molecule, including halogens
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in (6,1): # would we ever have hydrogen?
                marked.add(atom.GetIdx())

        #mark the four specific types of carbon atom
        for patt in PATT_TUPLE:
            for path in mol.GetSubstructMatches(patt):
                for atomindex in path:
                    marked.add(atomindex)
        #merge all connected marked atoms to a single FG
        groups = []
        while marked:
            grp = set([marked.pop()])
            merge(mol, marked, grp)
            groups.append(grp)

        #extract also connected unmarked carbon atoms
        ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
        ifgs = []
        for g in groups:
            uca = set()
            for atomidx in g:
                for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                    if n.GetAtomicNum() == 6:
                        uca.add(n.GetIdx())
            #ifgs.append(ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True), type=Chem.MolFragmentToSmiles(mol, g.union(uca),canonical=True)))
            ifgs.append(Chem.MolFragmentToSmiles(mol, g, canonical=True))
        return ifgs
    except:
        return None

def GetRingSystems(smi, includeSpiro=False):
    try:
        mol = Chem.MolFromSmiles(smi)
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon and (includeSpiro or nInCommon > 1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        ring_smi = []
        for system in systems:
            frag_smi = Chem.MolFragmentToSmiles(mol, system, canonical=True)
            ring_smi.append(frag_smi)
        return ring_smi
    except:
        return None