## GNN-Tox


[**To-Do List**](#quickstart-colab-in-the-cloud)
| [**Requirements**](#Requirements)
| [**Run**](#run)
| [**Experiments**](#Experiments)

**Pytorch implementation of paper "Predicting Molecule Toxicity via Graph Neural Networks" .**<br>

Authors: 

|  Name   | Group  | Email | TG |
|  ----   | ----   | ----  | ---- | 
| Xinze Li  | мНОД\_ИССА\_2020 | <sli_4@edu.hse.ru> | @lixinze


### To-Do list

- [x] Framework
- [x] Data splitting (random, scaffold)
- [ ] Models
	- [ ] baseline models 
        - [x] GCN, GAT, GIN
        - [ ] MPNN, Attentive FP 
    - [ ] ideas
        - [x] about modifying input graph, add super nodes for aromatic system (no improvement)
        - [ ]

### Requirements

```
pytorch                   1.9.1             
torch-geometric           2.0.2
rdkit                     2020.09.5
tqdm                      4.62.3
tensorboardx              2.4
```


### Run

```
python main.py 
```

### Experiments
- Idea 1: 1. Find all Aromatic atoms 2. Get aromatic systems 3. For each aromatic system add a super node that is connected with atoms in it by a directed edge. (For atoms connecting to substituents, add bidirectional edges) 4. During graph pooling, only consider about: all nodes (v1), all nodes except super nodes (v2), super nodes + heteroatoms + carbon pairs connected with multiple (double, triple) bonds (v3)

|     | tox21(AUC) |
|  ----   | ----  |
| GCN     | 84.46±0.37 |
| GCN_v1  | 84.18±0.56 |
| GCN_v2  | 83.99±0.71 |
| GCN_v3  | 82.65±0.54 |

Remark: no improvement
