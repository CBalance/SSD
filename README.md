# SSD

[[Homepage]()][[paper]()][[Poster]()]

official code for ""

![pipline](md-files/pipeline.png)

# Requirement

We use [Singularity](https://docs.sylabs.io/guides/3.3/user-guide/index.html) to build the enviroment. Download our enviroment: [excalibur.sif](https://portland-my.sharepoint.com/:u:/g/personal/wlin38-c_my_cityu_edu_hk/ESJUgH4yrsxPoZlOEfA9dCYBweBOif4vKVsBgRNqJH6E8Q?e=lWuBJH).
If you'd like to create environement yourself, the following python packages are required:
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
- using singularity: `singularity exec --bind  --nv path_to_excalibur.sif ./run.sh`
- using your own environment: `./run.sh`


# Citation



