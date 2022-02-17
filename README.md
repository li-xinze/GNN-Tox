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
- [ ] Data augmentations
    - [x] Node insertion (related to π-conjugated structure.) code: dataset/dataset_motif.py
    - [ ] -


### Requirements

```
pytorch                   1.9.1             
torch-geometric           2.0.2
rdkit                     2020.09.5
tqdm                      4.62.3
tensorboardx              2.4
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the other benchmarks for fine-tuning are saved in each folder.


### Run

Pre-training
```
python molclr.py
```

Fine-tuning
```
python finetune.py
```

### Experiments
Graph augmentation
- Node insertion: 1. Find all Aromatic atoms 2. Get aromatic systems 3. Remove all edges in aromatic systems 4. For each aromatic system add a super node that is connected with atoms in it by a directed edge. (For atoms connecting to substituents, add bidirectional edges) 
- Main code is in `dataset/dataset_motif.py`

Data splitting: random <br>
Metric: AUC

Remark: Finish by 20.02.2022
