import os

import numpy as np
import torch
import vapoursynth as vs
from torch.nn import functional as F

from .RIFE_HDv3 import Model


def RIFE(clip: vs.VideoNode, multi: int = 2, multi_den: int = 1, scale: float = 1.0, device_type: str = 'cuda', device_index: int = 0, fp16: bool = False) -> vs.VideoNode:
    '''
    RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    In order to avoid artifacts at scene changes, you should invoke `misc.SCDetect` on YUV or Gray format of the input beforehand so as to set frame properties.

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        multi: Framerate multiplier numerator.

        multi_den: Framerate multiplier denominator.

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
        raise vs.Error("RIFE: clip's number of frames must be at least 2")

    if not isinstance(multi, int):
        raise vs.Error('RIFE: multi must be integer')

    if not isinstance(multi_den, int):
        raise vs.Error('RIFE: multi_den must be integer')

    if multi < 2:
        raise vs.Error("RIFE: multi must be at least 2")

    if multi_den < 1:
        raise vs.Error("RIFE: multi_den must be at least 1")

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

    model = Model(device)
    model.load_model(os.path.dirname(__file__), -1)
    model.eval()

    w = clip.width
    h = clip.height
    tmp = max(128, int(128 / scale))
    pw = ((w - 1) // tmp + 1) * tmp
    ph = ((h - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    @torch.inference_mode()
    def rife(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if (n % multi_den != 0) or (n % multi == 0) or (n // multi == clip.num_frames - 1) or f[0].props.get('_SceneChangeNext'):
            return f[0]

        I0 = F.pad(frame_to_tensor(f[0]).to(device, non_blocking=True), padding)
        I1 = F.pad(frame_to_tensor(f[1]).to(device, non_blocking=True), padding)
        if fp16:
            I0 = I0.half()
            I1 = I1.half()

        output = model.inference(I0, I1, (n % multi) / multi, scale)
        return tensor_to_frame(output[:, :, :h, :w], f[0].copy())

    clip0 = vs.core.std.Interleave([clip] * multi)
    clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(frames=0)
    clip1 = vs.core.std.Interleave([clip1] * multi)
    clip2 = clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)
    if multi_den == 1:
        return clip2
    else:
        return clip2.std.SelectEvery(cycle = multi_den, offsets = 0)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f
