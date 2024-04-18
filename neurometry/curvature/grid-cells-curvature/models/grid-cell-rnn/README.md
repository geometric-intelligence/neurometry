# Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells
This repo contains the official implementation for the paper [Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells](https://arxiv.org/abs/2210.02684) (NeurReps Workshop 2022). 

Authors: Dehong Xu, Ruiqi Gao, Wen-Hao Zhang, Xue-Xin Wei, Ying Nian Wu

# 

Hexagon grid firing patterns emerge in our learned $v(x)$ from a 10-step RNN model: 

<img src="assets/rnn_pattern.png" alt="drawing" width="1000"/>

Learned hexagon grid patterns of $v(x)$, which is the hidden state vector in the LSTM transformation model: 

<img src="assets/lstm_pattern.png" alt="drawing" width="1000"/>

The learned model can perform accurate long distance path integration: 

<div align=center><img src="assets/path_integration.png" alt="drawing" width="700"/></div>

## Requirements
Requires python >= 3.5. To install dependencies:
```angular2
pip install -r requirements.txt
```

## Usage

- To train the nonlinear attractor model, run:

```angular2
python main.py --config=configs/rnn_isometry.py
```

- To train the LSTM, run:

```angular2
python main.py --config=configs/lstm_isometry.py
```

## Reference

```angular2
@article{xu2022conformal,
  title={Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells},
  author={Xu, Dehong and Gao, Ruiqi and Zhang, Wen-Hao and Wei, Xue-Xin and Wu, Ying Nian},
  journal={arXiv preprint arXiv:2210.02684},
  year={2022}
}
```
