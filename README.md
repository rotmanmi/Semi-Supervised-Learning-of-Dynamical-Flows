# Code for "Semi-supervised learning of partial differential operators and dynamical flows" -  UAI 2023

## Requirements
- [PyTorch 1.8.1](https://pytorch.org/)

## Files
The code is in the form of simple scripts. Each script shall be stand-alone and directly runnable.

- `hyper_1d.py` is the Fourier Neural Operator for 1D problem such as the 1D Burgers & Chafee-Infante equations.
- `hyper_2d.py` is the Fourier Neural Operator for 2D problem such as the 2D Burgers & Navier-Stokese quations.
- `hyper_3d.py` is the Fourier Neural Operator for 3D problem such as the 3D Navier-Stokes equation.

## Execution
### 1D Experiments:
The -i flag includes the $\mathcal{L}_{inter}$ term. --order flag is $P-1$. In order to use the GeLU variant add the --use_gelu flag.
For the regular Burgers:
```bash 
python hyper_1d.py -r -i --order 1 -d old --uniformintervals --use_tanh --epochs 500 --datapath $DATALOCATION
```
For the Chafee-Infante
```bash 
python hyper_1d.py -r -i --order 1 -d chafee --uniformintervals --use_tanh --epochs 500
```
For the Generalized Burgers, where the -g flag is  the order $q$ of the generalized burgers.
```bash
python hyper_1d.py -r -i --order 1 -d --epochs 500 -g 2 --uniformintervals --use_tanh --epochs 500 --datapath $DATALOCATION
```

### 2D Experiments
For the 2D burgers:
```bash
python hyper_2d.py -d burgers2d --order 1  --use_tanh --epochs 500 -i --uniformintervals --datapath $DATALOCATION
```
For the 2D Navier Stokes:
```bash
python hyper_2d.py -d navier --order 1  --use_tanh --epochs 500 -i --uniformintervals --datapath $DATALOCATION
```

### 3D Experiments
```bash
python hyper_3d.py -d navier --order 0  --use_tanh --epochs 500 -i --uniformintervals --datapath $DATALOCATION
```


## Datasets
The 1D dataset (used for Burgers 1D, Generalized Burgers 1D & Chafee-Infante) may be obtained from [1D dataset](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing), Burgers_R10.zip.



## Dataset Generation

### 2D Burgers
```bash
    python generate_2d_burgers_data.py
```
### 2D Navier Stokes
```bash
    python generate_2dnavier_data.py
```
### 3D Navier Stokes
```
    python generate_3dnavier_data.py
```
