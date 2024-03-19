## Introduction
This is **[AMS: Attention-Base Multiscale Networks for Image-Text Matching]**, source code of AMS.

![image](https://github.com/WMSlalalala/AMS/blob/main/fig5.png)

## Requirements and Installation
We recommended the following dependencies.
* Python 3.6
* [PyTorch](http://pytorch.org/) 2.1.1
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Download data
The raw images can be downloaded from their original sources [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/). We will use the [json](https://drive.google.com/drive/folders/1mondFS6TCbzvz2ZUk4UDFAxnD8xk25ie?usp=drive_link) files to leverage these data.
We refer to the path of extracted files as `$DATA_PATH`. by(https://github.com/LuminosityX/FNE) 

## Train new models
Run `train.py`:
For AMS on Flickr30K:
```bash
python train.py --data_path $DATA_PATH/f30k/images --dataset flickr --max_violation --max_violation
```

## Evaluate trained models
Test on Flickr30K:
```bash
python train.py --data_path $DATA_PATH/f30k/images  --resume /runs/runx/model_best.pth.tar  --dataset flickr --max_violation --test
```

## Reference
```
@inproceedings{zhang2022negative,
  title={Negative-Aware Attention Framework for Image-Text Matching},
  author={Zhang, Kun and Mao, Zhendong and Wang, Quan and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15661--15670},
  year={2022}
}
@inproceedings{li2023your,
  title={Your negative may not be true negative: Boosting image-text matching with false negative elimination},
  author={Li, Haoxuan and Bin, Yi and Liao, Junrong and Yang, Yang and Shen, Heng Tao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={924--934},
  year={2023}
}

```
```
@article{zhang2022unified,
  title={Unified adaptive relevance distinguishable attention network for image-text matching},
  author={Zhang, Kun and Mao, Zhendong and Liu, An-An and Zhang, Yongdong},
  journal={IEEE Transactions on Multimedia},
  volume={25},
  pages={1320--1332},
  year={2022},
  publisher={IEEE}
}
```
