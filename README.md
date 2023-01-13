# Dialog-HGAT
This repository contains the Pytorch implementation of [Dialogue Relation Extraction with Document-level Heterogeneous Graph Attention Networks](https://arxiv.org/pdf/2009.05092.pdf).

## Setup
Download GloVe vectors from [here](https://www.kaggle.com/thanakomsn/glove6b300dtxt/data) and put it into `dataset/` folder

Next Install the required libraries:
1. Assume you have installed Pytorch >= 1.5
2. Install dgl library according to your cuda version using the commands below
```sh
pip install --pre dgl-cu100     # For CUDA 10.0 Build
pip install --pre dgl-cu101     # For CUDA 10.1 Build
pip install --pre dgl-cu102     # For CUDA 10.2 Build
```
3. Install PytorchLightning [github](https://github.com/PyTorchLightning/pytorch-lightning)
4. Install from requirements.txt by `pip install -r requirements.txt` and run `python -m spacy download en_core_web_sm`

## Run code

### Training
```sh
python main.py
```

### Testing
```sh
python main.py --mode test --ckpt_path [your_ckpt_file_path]
```

## Citation
If you find the code helpful in your research, please cite:

```
@article{chen2020dialogue,
  title={Dialogue relation extraction with document-level heterogeneous graph attention networks},
  author={Chen, Hui and Hong, Pengfei and Han, Wei and Majumder, Navonil and Poria, Soujanya},
  journal={Cognitive Computation},
  year={2022}
}
```
