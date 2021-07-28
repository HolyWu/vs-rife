# RIFE
RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

Ported from https://github.com/hzwer/arXiv2020-RIFE


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


## Installation
`pip install --upgrade vsrife`


## Usage
```python
from vsrife import RIFE

ret = RIFE(clip)
```

See `__init__.py` for the description of the parameters.
