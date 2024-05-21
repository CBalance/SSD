# SSD: Learning Spatial Similarity Distribution for Few-shot Object Counting

[[Homepage](https://github.com/CBalance/SSD)][[paper](https://paperswithcode.com/paper/learning-spatial-similarity-distribution-for)]

official code for ijcai24 "Learning Spatial Similarity Distribution for Few-shot Object Counting"

![pipline](md-files/pipeline.png)

# Requirement
The following python packages are required:
```
pytorch == 1.9.0
torchvision == 0.10.0
mmcv == 1.3.13
timm == 0.4.12
termcolor
yacs
einops
```

# Data Preparation

- Download [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
- modify the `root` in line 12 of `datasets/gendata384x576.py` to the local path of FSC-147.
- running the file `datasets/gendata384x576.py`

# Training

- modify the `datapath` in `run.sh` to the local path of FSC-147 dataset
- using your own environment: `./run.sh`


# Citation



