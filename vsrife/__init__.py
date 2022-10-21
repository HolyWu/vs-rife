from __future__ import annotations

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

dir_name = osp.dirname(__file__)


def RIFE(
    clip: vs.VideoNode,
    device_type: str = 'cuda',
    device_index: int = 0,
    fp16: bool = False,
    model: str = '4.6',
    multi: int = 2,
    scale: float = 1.0,
    ensemble: bool = False,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:            Clip to process. Only RGBS format is supported.
    :param device_type:     Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.
    :param device_index:    Device ordinal for the device type.
    :param fp16:            Enable FP16 mode.
    :param model:           Model version to use. Must be '4.0', '4.1', '4.2', '4.3', '4.4', '4.5', or '4.6'.
    :param multi:           Multiple of the frame counts.
    :param scale:           Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                            Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:        Smooth predictions in areas where the estimation is uncertain.
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

    if model not in ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6']:
        raise vs.Error("RIFE: model must be '4.0', '4.1', '4.2', '4.3', '4.4', '4.5', or '4.6'")

    if not isinstance(multi, int):
        raise vs.Error('RIFE: multi must be integer')

    if multi < 2:
        raise vs.Error("RIFE: multi must be at least 2")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error('RIFE: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0')

    if osp.getsize(osp.join(dir_name, 'flownet_v4.0.pkl')) == 0:
        raise vs.Error("RIFE: model files have not been downloaded. run 'python -m vsrife' first")

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    match model:
        case '4.0':
            from .IFNet_HDv3_v4_0 import IFNet
        case '4.1':
            from .IFNet_HDv3_v4_1 import IFNet
        case '4.2':
            from .IFNet_HDv3_v4_2 import IFNet
        case '4.3':
            from .IFNet_HDv3_v4_3 import IFNet
        case '4.4':
            from .IFNet_HDv3_v4_4 import IFNet
        case '4.5':
            from .IFNet_HDv3_v4_5 import IFNet
        case '4.6':
            from .IFNet_HDv3_v4_6 import IFNet

    checkpoint = torch.load(osp.join(dir_name, f'flownet_v{model}.pkl'), map_location='cpu')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items() if 'module.' in k}

    flownet = IFNet(device, scale, ensemble)
    flownet.load_state_dict(checkpoint, strict=False)
    flownet.eval()
    flownet.to(device)

    w = clip.width
    h = clip.height
    tmp = max(128, int(128 / scale))
    pw = ((w - 1) // tmp + 1) * tmp
    ph = ((h - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    @torch.inference_mode()
    def rife(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        if (n % multi == 0) or (n // multi == clip.num_frames - 1) or f[0].props.get('_SceneChangeNext'):
            return f[0]

        img0 = frame_to_tensor(f[0]).to(device)
        img1 = frame_to_tensor(f[1]).to(device)
        if fp16:
            img0 = img0.half()
            img1 = img1.half()
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        imgs = torch.cat((img0, img1), dim=1)
        timestep = torch.full((1, 1, imgs.shape[2], imgs.shape[3]), fill_value=(n % multi) / multi, device=device)
        output = flownet(imgs, timestep)
        return tensor_to_frame(output[:, :, :h, :w], f[0].copy())

    clip0 = vs.core.std.Interleave([clip] * multi)
    clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(frames=0)
    clip1 = vs.core.std.Interleave([clip1] * multi)
    return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)


def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame
