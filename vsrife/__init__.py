import numpy as np
import os
import torch
import vapoursynth as vs
from torch.nn import functional as F

core = vs.core


def RIFE(clip: vs.VideoNode, model_ver: float=3.5, scale: float=1.0, device_type: str='cuda', device_index: int=0, fp16: bool=False) -> vs.VideoNode:
    '''
    RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    In order to avoid artifacts at scene change, you should invoke `misc.SCDetect` on YUV or Gray format of the input beforehand so as to set frame properties.

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        model_ver: Model version to use. Must be 3.1, 3.5, or 3.8.

        scale: Controls the process resolution for optical flow model. Try scale=0.5 for 4K video. Must be 0.25, 0.5, 1.0, 2.0, or 4.0.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RIFE: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RIFE: only RGBS format is supported')

    if clip.num_frames < 2:
        raise vs.Error('RIFE: number of frames must be at least 2')

    if model_ver not in [3.1, 3.5, 3.8]:
        raise vs.Error('RIFE: model_ver must be 3.1, 3.5, or 3.8')

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error('RIFE: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("RIFE: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('RIFE: CUDA is not available')

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if model_ver == 3.1:
        from .model31.RIFE_HDv3 import Model
        model_dir = 'model31'
    elif model_ver == 3.5:
        from .model35.RIFE_HDv3 import Model
        model_dir = 'model35'
    else:
        from .model38.RIFE_HDv3 import Model
        model_dir = 'model38'

    model = Model(device, fp16)
    model.load_model(os.path.join(os.path.dirname(__file__), model_dir), -1)
    model.eval()
    model.device()

    w = clip.width
    h = clip.height
    tmp = max(32, int(32 / scale))
    pw = ((w - 1) // tmp + 1) * tmp
    ph = ((h - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    clip0 = core.std.Interleave([clip, clip])
    clip1 = clip0.std.DuplicateFrames(frames=clip0.num_frames - 1).std.DeleteFrames(frames=0)

    def rife(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if not (n & 1) or n == clip0.num_frames - 1 or f[0].props.get('_SceneChangeNext'):
            return f[0]

        I0 = F.pad(frame_to_tensor(f[0]).to(device, non_blocking=True), padding)
        I1 = F.pad(frame_to_tensor(f[1]).to(device, non_blocking=True), padding)
        if fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.no_grad():
            middle = model.inference(I0, I1, scale)

        return tensor_to_frame(middle[:, :, :h, :w], f[0])

    return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane)) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.data.squeeze().cpu().numpy()
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane)), arr[plane, :, :])
    return fout
