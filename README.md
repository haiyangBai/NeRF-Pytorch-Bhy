## The code for NeRF completed with PyTorch and PyTorch-lighting, respectively.

### - Environment
- Ubuntu 18.04
- Python 3.7
- CUDA 11.x
- Pytorch 1.9.1
- Pytorch-Lightning 1.6.4

### - Install via Anaconda
```
$ conda create -n NeRF python=3.8
$ conda activate NeRF
$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
```

### - Dataset

By running
```
bash data_download.sh
```

 ### - Training
 - Classical training pipeline by running   `bash train_classical.sh`
 - Packing training pipeline by running `bash train_packing.sh`
 - Pytorch Linghting version by running `bash train_classical.sh`

### Acknowledgement
Our initial code was borrowed from 
- [nerf-pl:https://github.com/kwea123/nerf_pl](https://github.com/kwea123/nerf_pl)
- [CodeNeRF:https://github.com/wbjang/code-nerf.git](https://github.com/wbjang/code-nerf.git)