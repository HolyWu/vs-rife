# RIFE
Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on https://github.com/hzwer/Practical-RIFE.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0.dev20241023 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.4.0 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0.dev20241023 or later

To install the latest nightly build of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install --pre -U torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --no-deps --pre -U torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -U tensorrt-cu12 tensorrt-cu12_bindings tensorrt-cu12_libs --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsrife
```

If you want to download all models at once, run `python -m vsrife`.
If you prefer to only download the model you specified at first run, set `auto_download=True` in `rife()`.


## Usage
```python
from vsrife import rife

ret = rife(clip)
```

See `__init__.py` for the description of the parameters.
