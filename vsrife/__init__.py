import os

import numpy as np
import torch
import vapoursynth as vs
from torch.nn import functional as F

from .RIFE_HDv3 import Model


def RIFE(
    clip: vs.VideoNode,
    device_type: str = 'cuda',
    device_index: int = 0,
    fp16: bool = False,
    multi: int = 2,
    scale: float = 1.0,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:            Clip to process. Only RGBS format is supported.
    :param device_type:     Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.
    :param device_index:    Device ordinal for the device type.
    :param fp16:            Enable FP16 mode.
    :param multi:           Multiple of the frame counts.
    :param scale:           Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                            Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RIFE: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RIFE: only RGBS format is supported')

    if clip.num_frames < 2:
        raise vs.Error("RIFE: clip's number of frames must be at least 2")

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("RIFE: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('RIFE: CUDA is not available')

    if not isinstance(multi, int):
        raise vs.Error('RIFE: multi must be integer')

    if multi < 2:
        raise vs.Error("RIFE: multi must be at least 2")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error('RIFE: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0')

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
        if (n % multi == 0) or (n // multi == clip.num_frames - 1) or f[0].props.get('_SceneChangeNext'):
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
    return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f
