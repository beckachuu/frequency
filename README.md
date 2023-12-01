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


## Run experiment

### Run

```sh
python main.py
```

### View Result
Images are saved in `\output\{Your input folder name}\{low or high frequency}_{r}\`:
- To view detected images: open folder `\detects\` inside the above folder


## References

- The code obtaining low and high frequency component is proposed by Wang et al.
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
