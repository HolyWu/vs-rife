# RIFE
Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on https://github.com/hzwer/Practical-RIFE.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.4.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)

`trt` requires additional Python packages:
- [TensorRT](https://developer.nvidia.com/tensorrt/) 10.0.1
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.4.0.dev

To install TensorRT, run `pip install tensorrt==10.0.1 tensorrt-cu12_bindings==10.0.1 tensorrt-cu12_libs==10.0.1 --extra-index-url https://pypi.nvidia.com`

To install Torch-TensorRT, Windows users can pip install the whl file on [Releases](https://github.com/HolyWu/vs-rife/releases). Linux users can run `pip install --pre torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu121` (requires PyTorch nightly build).


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
