# HamDiv
`diversity.py`provides an implementation of **Hamiltonian diversity**, together with all existing molecular diversity metrics.

## Dependencies
- numpy
- rdkit
- tqdm
- networkx
- python_tsp

## Usage Example

This example is corresponding to the example of Figure 2 and Table 2 in the paper.

```python
from diversity import dist_array, diversity_all, HamDiv

smiles = ['Cc1cc(C(O)CNC2(C)CC2S)ccc1O', 'CCc1cc(C(O)CNC(C)(C)C)ccc1O', 'CCCC(C)NCC(O)c1ccc(O)c(CO)c1', 'CNCC(S)c1ccc(O)c(CO)c1', 'CNCC(C)(C)C(O)c1ccc(S)c(CO)c1']

Richness = diversity_all(smiles=smiles, mode="Richness")
print(Richness)
IntDiv = diversity_all(smiles=smiles, mode="IntDiv")
print(IntDiv)
Circles = diversity_all(smiles=smiles, mode="NCircles-0.7") # 0.7 is an adjustable hyper-parameter
print(Circles)
HamDiv = diversity_all(smiles=smiles, mode="HamDiv")
HamDiv_same = HamDiv(smiles=smiles)
print(HamDiv, HamDiv_same)
```

`additional_term.py` provides the core code of incorporating Hamiltonian diversity into the scoring function for molecular generation in Section 4.2 in the paper.
