# HamDiv
An implementation of Hamiltonian diversity, together with other existing molecular diversity metrics.

## Dependencies
- numpy
- rdkit
- tqdm

## Usage Example
```python
smiles = ['Cc1cc(C(O)CNC2(C)CC2S)ccc1O', 'CNCC(S)c1ccc(O)c(CO)c1']
Richness = diversity_all(smiles=smiles, mode="Richness")
IntDiv = diversity_all(smiles=smiles, mode="IntDiv")
Circles = diversity_all(smiles=smiles, mode="NCircles-0.7") # 0.7 is the hyper-parameter
HamDiv = diversity_all(smiles=smiles, mode="HamDiv")
```
