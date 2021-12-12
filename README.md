# RIFE
RIFE function for VapourSynth, based on https://github.com/hzwer/Practical-RIFE.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchaudio` is not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


## Installation
```
pip install --upgrade vsrife
```


## Usage
```python
from vsrife import RIFE

ret = RIFE(clip)
```

See `__init__.py` for the description of the parameters.
