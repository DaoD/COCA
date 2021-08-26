# Contrastive Learning of User Behavior Sequence for Context-Aware Document Ranking

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Abstract
This repository contains the source code and datasets for the CIKM 2021 paper [Contrastive Learning of User Behavior Sequence for Context-Aware Document Ranking](https://arxiv.org/pdf/2108.10510.pdf) by Zhu et al. <br>

Context information in search sessions has proven to be useful for capturing user search intent. Existing studies explored user behavior sequences in sessions in different ways to enhance query suggestion or document ranking. However, a user behavior sequence has often been viewed as a definite and exact signal reflecting a user's behavior. In reality, it is highly variable: user's queries for the same intent can vary, and different documents can be clicked. To learn a more robust representation of the user behavior sequence, we propose a method based on contrastive learning, which takes into account the possible variations in user's behavior sequences.  Specifically, we propose three data augmentation strategies to generate similar variants of user behavior sequences and contrast them with other sequences. In so doing, the model is forced to be more robust regarding the possible variations. The optimized sequence representation is incorporated into  document ranking. Experiments on two real query log datasets show that our proposed model outperforms the state-of-the-art methods significantly, which demonstrates the effectiveness of our method for context-aware document ranking.

Authors: Yutao Zhu, Jian-Yun Nie, Zhicheng Dou, Zhengyi Ma, Xinyu Zhang, Pan Du, Xiaochen Zuo, and Hao Jiang

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.8.5 <br>
- Pytorch 1.8.1 (with GPU support)<br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- Download the data. 
  - For AOL dataset, please contact the author of [CARS](https://arxiv.org/pdf/1906.02329.pdf)
  - For Tiangong dataset, we preprocess it and you can download it from the [link]()
- Unzip the data

### Contrastive Learning Stage
#### AOL Dataset
```
python runBertContras.py --task aol --bert_model_path ../BERT/BertModel/
```

#### Tiangong Dataset
```
python runBertContras.py --task tiangong --bert_model_path ../BERT/BertChinese/
```

We will share the model after contrastive learning as soon as possible!

### Ranking Stage


The diarectory structure is:
```
COCA
├── BERT
│   ├── BERTChinese
│   └── BERTModel
├── ContrastiveLearning
│   ├── BertContrasPretrain.py
│   ├── data
│   │   ├── aol
│   │   └── tiangong
│   │       ├── dev.pos.txt
│   │       ├── test.pos.txt
│   │       └── train.pos.txt
│   ├── file_preprocess_dataset.py
│   ├── log
│   ├── model
│   ├── output
│   │   ├── aol
│   │   └── tiangong
│   └── runBertContras.py
└── Ranking
    ├── BertSessionSearch.py
    ├── Trec_Metrics.py
    ├── data
    │   ├── aol
    │   └── tiangong
    │       ├── dev.point.txt
    │       ├── test.point.lastq.txt
    │       ├── test.point.preq.txt
    │       └── train.point.txt
    ├── file_dataset.py
    ├── log
    ├── model
    ├── output
    │   ├── aol
    │   └── tiangong
    └── runBert.py
```

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{ZhuNZDJD21,
  author    = {Yutao Zhu and
               Jian{-}Yun Nie and
               Zhicheng Dou and
               Zhengyi Ma and
               Xinyu Zhang and
               Pan Du and
               Xiaochen Zuo and
               Hao Jiang},
  title     = {Contrastive Learning of User Behavior Sequence for Context-Aware Document Ranking},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, QLD, Australia, November 1-5, 2021},
  publisher = {{ACM}},
  year      = {2021}
}
```
