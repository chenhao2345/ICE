# ICE
This is the official PyTorch implementation of the ICCV 2021 paper
[ICE: Inter-instance Contrastive Encoding for Unsupervised Person
Re-identification](https://arxiv.org/pdf/2103.16364.pdf).

[[Video](https://drive.google.com/file/d/1E__ru9u_oRcb44-WIH_GjBTv1-_5rcO2/view?usp=sharing)]   [[Poster](https://drive.google.com/file/d/1HEkgtUCSOixIndH1ClhRZfAQGTIFfY-n/view?usp=sharing)]

![teaser](figs/figure8.png)

## Installation

```shell
git clone https://github.com/chenhao2345/ICE
cd ICE
python setup.py develop
```

## Prepare Datasets

Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
ICE/examples/data
├── dukemtmc-reid
│   └── DukeMTMC-reID
├── market1501
└── msmt17
    └── MSMT17_V1(or MSMT17_V2)
```

## Training
We used **4 GPUs** to train our model.
 
Train [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf):
```
python examples/unsupervised_train.py --dataset-target market1501
```
Train [DukeMTMC-reID](https://arxiv.org/abs/1609.01775):
```
python examples/unsupervised_train.py --dataset-target dukemtmc-reid
```
Train [MSMT17](https://arxiv.org/abs/1711.08565):
```
python examples/unsupervised_train.py --dataset-target msmt17
```
## Citation
If you find this project useful, please kindly star our project and cite our paper.
```bibtex
@InProceedings{Chen_2021_ICCV,
    author    = {Chen, Hao and Lagadec, Benoit and Bremond, Fran\c{c}ois},
    title     = {ICE: Inter-Instance Contrastive Encoding for Unsupervised Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14960-14969}
}
```

