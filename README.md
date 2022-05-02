## GNN-Tox


| [**Requirements**](#Requirements)
| [**Run**](#run)

**Pytorch implementation of paper "Predicting Molecule Toxicity via Graph Neural Networks" .**<br>

Authors: 

|  Name   | Group  | Email | TG |
|  ----   | ----   | ----  | ---- | 
| Xinze Li  | мНОД\_ИССА\_2020 | <sli_4@edu.hse.ru> | @lixinze


### Requirements

```
pytorch                   1.9.1             
torch-geometric           2.0.2
rdkit                     2020.09.5
tqdm                      4.62.3
tensorboardx              2.4
```


### Run

Pre-training
- Attribute Masking: `python pretrain_masking.py`
- Motif/DC label prediction: `python pretrain_motif.py`
- DGSSL: `python pretrain_dgssl.py`


Fine-tuning
```
python finetune.py
```
or use script `finetune.sh`


