Multi-Anchor Active Domain Adaptation for Semantic Segmentation

Munan Ning*, Donghuan Lu*, Dong Wei†, Cheng Bian, Chenglang Yuan, Shuang Yu, Kai Ma, Yefeng Zheng

[paper](https://arxiv.org/abs/2108.08012)


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Notes](#note)

## Introduction

This respository contains the MADA method as described in the ICCV 2021 Oral paper ["Multi-Anchor Active Domain Adaptation for Semantic Segmentation"](https://arxiv.org/abs/2108.08012).

## Requirements

The code requires Pytorch >= 0.4.1 with python 3.6. The code is trained using a NVIDIA Tesla V100 with 32 GB memory. You can simply reduce the batch size in stage 2 to run on a smaller memory.

## Usage

1. Preparation:
* Download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset as the source domain, and the [Cityscapes](https://www.cityscapes-dataset.com/) dataset as the target domain.
* Download the [weights](https://drive.google.com/drive/folders/16DxIRuo06afRp2FpXvWg9nAeYiTL25yP?usp=sharing) and [features](https://drive.google.com/drive/folders/1ybhJUTh7y1QcOnpNz0jnxkpz5TTtO55s?usp=sharing). Move features to the MADA directory.

2. Setup the config files.
* Set the data paths
* Set the pretrained model paths

3. Training-
* To run the code:
~~~~
python train.py
~~~~
* During the training, the generated files (log file) will be written in the folder 'runs/..'.

4. Evaluation
* Set the config file for test (configs/test_from_city_to_gta.yml):
* Run:
~~~~
python3 test.py
~~~~
to see the results.

4. Constructing anchors
* Setting the config file 'configs/CAC_from_gta_to_city.yml' as illustrated before.
* Run:
~~~~
python cac.py
~~~~
* The anchor file would be in 'run/cac_from_gta_to_city/..'





## License

[MIT](LICENSE)

The code is heavily borrowed from the CAG_UDA (https://github.com/RogerZhangzz/CAG_UDA).

If you use this code and find it usefule, please cite:
~~~~
@inproceedings{ning2021multi,
  title={Multi-Anchor Active Domain Adaptation for Semantic Segmentation},
  author={Ning, Munan and Lu, Donghuan and Wei, Dong and Bian, Cheng and Yuan, Chenglang and Yu, Shuang and Ma, Kai and Zheng, Yefeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9112--9122},
  year={2021}
}
~~~~

## Notes
The anchors are calcuated based on features captured by decoders.

In this paper, we utilize the more powerful decoder in DeeplabV3+, it may cause somewhere unfair. So we strongly recommend the [ProDA](https://github.com/microsoft/ProDA) which utilize origin DeeplabV2 decoder.
