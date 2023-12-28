# Frequency Filter

## Description

This project aims to study the intriguing characteristics of images when divided into high-frequency and low-frequency components.


## Table of Contents

- [Getting Started](#getting-started)

  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Input Configuration](#input-configuration)
- [Run experiment](#run-experiment)
  - [Run](#run)
  - [View Result](#view-result)
- [References](#references)

## Getting started

### Prerequisites

- Python

### Installation

```sh
git clone https://github.com/beckachuu/FrequencyFilter.git
cd FrequencyFilter
pip install -r requirements.txt
```

**Note**: You may optionally wish to create a [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to prevent conflicts with your system's Python environment.

### Data Preparation

#### `Image`:

- [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)

#### `Groundtruth data`:

- [COCO 2017 Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)


### Input Configuration

General input required arguments are in [config.ini](config.ini).


## Run analysis and experiments

### Run

#### Generate images

- Run analyze: separate high and low frequency components of input images
```sh
python analyze_freq.py
```

- Run experiments: run a chosen experiment
```sh
python run_exp.py
```

#### Detect objects

- Detect objects for experiment results
```sh
python detect.py
```

- Plot detected bboxes for experiment results
```sh
python plot_exp_demo.py
```


#### Evaluate

- Evaluate mean Average Precision (mAP) for experiment results
```sh
python eval.py
```

#### Extra

- Experiment with kernel: analyze magnitude and phase of different types of kernels in Fourier domain
```sh
python kernel_exp.py
```


### View Result

- `\output\{Your input folder name}\analyze\`: analyze results
- `\output\{Your input folder name}\EXP_{#}\`: experiments results
- Detected images: open folder `\detects\` inside each result folder


## References

- The code obtaining low and high frequency component is partially obtained from the repository of Wang et al.
```
@article{Wang2019HighFrequencyCH,
  title={High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks},
  author={Haohan Wang and Xindi Wu and Pengcheng Yin and Eric P. Xing},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
  pages={8681-8691},
  url={https://api.semanticscholar.org/CorpusID:173188317}
}
```


<!-- **Developers**: Caugiay dist., Hanoi, Vietnam. -->

<!-- If you have any questions, please contact via Vu Ha Minh Trang <20020267@vnu.edu.vn> . -->

This project welcomes contributions and suggestions.
