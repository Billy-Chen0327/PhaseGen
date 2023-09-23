# PhaseGen: Deep Generative Model for synthesizing seismic waveforms with P- and S- arrival labels

![diagram](./diagram.jpg)

By Guoyi Chen @ USTC, Email: billychen@mail.ustc.edu.cn

## 1. Install (using conda)

```
conda create -n PhaseGen python=3.8
conda activate PhaseGen
pip install -r requirements.txt
```

## 2. Demo for synthesizing waveforms

Demo script using jupyter notebook are prepared to give description on synthesizing waveforms, which are located in directory: `demo/demo.ipynb`

## 3. Training

Training code is located in directory: **train/**

The training parameter can be set by **config.py**

Training dataset is located in: **train/data/ ** and saved as npz format.

##  Citation

If you find PhaseGen useful, please cite the following reference:

Guoyi Chen, Junlun Li and Hao Guo; Deep generative model conditioned by phase picks for synthesizing labeled seismic waveforms with limited data. *arXiv preprint*, 2023; doi: [ https://doi.org/10.48550/arXiv.2309.11297](https://doi.org/10.48550/arXiv.2309.11297)

## License

The **PhaseGen** package is distributed under the `MIT license` (free software).