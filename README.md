# CubeNet: Array-based seismic phase picking with Deep Learning

A multi-trace seismic phase picking method for both P- and S- arrivals which is particularly suitable for dense array.

By Guoyi Chen @ USTC, Email: billychen@mail.ustc.edu.cn

## 1. Using conda

```
conda create -n CubeNet python=3.6
conda activate CubeNet
pip install -r requirements.txt
```

## 2. Prediction Demo

Located in directory: Demo/

**Demo1**: Predict P/S arrival phases with given raw waveforms and station locations

**Demo2**: Direct Pick P/S arrivals from a given cube

**Demo3**: Usage of data resampling in CubeNet

## 3. Parameter Setup

Some parameters (like GPU acceleration) can be set by **config.py** in the CubeNet package.

A GPU with more than 4GB of memory is recommended to run CubeNet.

## License

The **CubeNet** package is distributed under the `MIT license` (free software).