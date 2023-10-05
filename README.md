# RIFE
Real-Time Intermediate Flow Estimation for Video Frame Interpolation, based on https://github.com/hzwer/Practical-RIFE.


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.0.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later

`trt` requires additional runtime libraries:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12.1
- [cuDNN](https://developer.nvidia.com/cudnn) 8.9
- [TensorRT](https://developer.nvidia.com/tensorrt) 8.6.1

For ease of installation on Windows, you can download the 7z file on [Releases](https://github.com/HolyWu/vs-rife/releases) which contains required runtime libraries and Python wheel file. Either add the unzipped directory to your system `PATH` or copy the DLL files to a directory which is already in your system `PATH`. Finally pip install the Python wheel file.


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
