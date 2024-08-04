# RIFE
Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on https://github.com/hzwer/Practical-RIFE.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.5.0.dev
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)

`trt` requires additional Python packages:
- [TensorRT](https://developer.nvidia.com/tensorrt/) 10.1.0
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.5.0.dev

To install or update PyTorch and Torch-TensorRT to the latest nightly build, run `pip install --pre -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124 --extra-index-url https://pypi.nvidia.com`


## Installation
```
pip install -U vsrife
python -m vsrife
```


## Usage
```python
from vsrife import rife

ret = rife(clip)
```

See `__init__.py` for the description of the parameters.
