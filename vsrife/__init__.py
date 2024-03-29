from __future__ import annotations

import os
from fractions import Fraction
from threading import Lock

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision

__version__ = "4.2.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

models = [
    "4.0",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "4.10",
    "4.11",
    "4.12",
    "4.12.lite",
    "4.13",
    "4.13.lite",
    "4.14"
    "4.14.lite"
    "4.15"
    "4.15.lite"
]

models_str = ""
for model in models:
    models_str += "'" + model + "', "
models_str = models_str[:-2]


@torch.inference_mode()
def rife(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 2,
    trt: bool = False,
    trt_max_workspace_size: int = 1 << 30,
    trt_cache_path: str = model_dir,
    model: str = "4.6",
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = True,
    sc_threshold: float | None = None,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param trt:                     Use TensorRT for high-performance inference.
                                    Not supported for '4.0' and '4.1' models.
    :param trt_max_workspace_size:  Maximum workspace size for TensorRT engine.
    :param trt_cache_path:          Path for TensorRT engine file. Engine will be cached when it's built for the first
                                    time. Note each engine is created for specific settings such as model path/name,
                                    precision, workspace etc, and specific GPUs and it's not portable.
    :param model:                   Model version to use.
    :param factor_num:              Numerator of factor for target frame rate.
                                    For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param factor_den:              Denominator of factor for target frame rate.
    :param fps_num:                 Numerator of target frame rate. Override `factor_num` and `factor_den` if specified.
    :param fps_den:                 Denominator of target frame rate.
    :param scale:                   Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                                    Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:                Smooth predictions in areas where the estimation is uncertain.
    :param sc:                      Avoid interpolating frames over scene changes.
    :param sc_threshold:            Threshold for scene change detection. Must be between 0.0 and 1.0.
                                    Leave it None if the clip already has _SceneChangeNext properly set.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("rife: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("rife: only RGBH and RGBS formats are supported")

    if clip.num_frames < 2:
        raise vs.Error("rife: clip's number of frames must be at least 2")

    if not torch.cuda.is_available():
        raise vs.Error("rife: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("rife: num_streams must be at least 1")

    if model not in models:
        raise vs.Error(f"rife: model must be {models_str}")

    if factor_num < 1:
        raise vs.Error("rife: factor_num must be at least 1")

    if factor_den < 1:
        raise vs.Error("rife: factor_den must be at least 1")

    if fps_num is not None and fps_num < 1:
        raise vs.Error("rife: fps_num must be at least 1")

    if fps_den is not None and fps_den < 1:
        raise vs.Error("rife: fps_den must be at least 1")

    if fps_num is not None and fps_den is not None and clip.fps == 0:
        raise vs.Error("rife: clip does not have a valid frame rate and hence fps_num and fps_den cannot be used")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("rife: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    if os.path.getsize(os.path.join(model_dir, "flownet_v4.0.pkl")) == 0:
        raise vs.Error("rife: model files have not been downloaded. run 'python -m vsrife' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    if fp16:
        torch.set_default_dtype(torch.half)

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case "4.0":
            from .IFNet_HDv3_v4_0 import IFNet
        case "4.1":
            from .IFNet_HDv3_v4_1 import IFNet
        case "4.2":
            from .IFNet_HDv3_v4_2 import IFNet
        case "4.3":
            from .IFNet_HDv3_v4_3 import IFNet
        case "4.4":
            from .IFNet_HDv3_v4_4 import IFNet
        case "4.5":
            from .IFNet_HDv3_v4_5 import IFNet
        case "4.6":
            from .IFNet_HDv3_v4_6 import IFNet
        case "4.7":
            from .IFNet_HDv3_v4_7 import IFNet
        case "4.8":
            from .IFNet_HDv3_v4_8 import IFNet
        case "4.9":
            from .IFNet_HDv3_v4_9 import IFNet
        case "4.10":
            from .IFNet_HDv3_v4_10 import IFNet
        case "4.11":
            from .IFNet_HDv3_v4_11 import IFNet
        case "4.12":
            from .IFNet_HDv3_v4_12 import IFNet
        case "4.12.lite":
            from .IFNet_HDv3_v4_12_lite import IFNet
        case "4.13":
            from .IFNet_HDv3_v4_13 import IFNet
        case "4.13.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet
        case "4.14":
            from .IFNet_HDv3_v4_13_lite import IFNet
        case "4.14.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet
        case "4.15":
            from .IFNet_HDv3_v4_13_lite import IFNet
        case "4.15.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet
            
    model_name = f"flownet_v{model}.pkl"

    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False)
    flownet.eval().to(device, memory_format=torch.channels_last)

    w = clip.width
    h = clip.height
    tmp = max(128, int(128 / scale))
    pw = ((w - 1) // tmp + 1) * tmp
    ph = ((h - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    if trt:
        device_name = torch.cuda.get_device_name(device)
        trt_version = tensorrt.__version__
        dimensions = f"{pw}x{ph}"
        precision = "fp16" if fp16 else "fp32"
        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_path),
            (
                f"{model_name}"
                + f"_{device_name}"
                + f"_trt-{trt_version}"
                + f"_{dimensions}"
                + f"_{precision}"
                + f"_workspace-{trt_max_workspace_size}"
                + f"_scale-{scale}"
                + f"_ensemble-{ensemble}"
                + ".pt"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            lower_setting = LowerSetting(
                lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
                min_acc_module_size=1,
                max_workspace_size=trt_max_workspace_size,
                dynamic_batch=False,
                tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
                | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
            )
            lowerer = Lowerer.create(lower_setting=lower_setting)
            flownet = lowerer(
                flownet,
                [
                    torch.zeros((1, 3, ph, pw), device=device).to(memory_format=torch.channels_last),
                    torch.zeros((1, 3, ph, pw), device=device).to(memory_format=torch.channels_last),
                    torch.zeros((1, 1, ph, pw), device=device).to(memory_format=torch.channels_last),
                ],
            )
            torch.save(flownet, trt_engine_path)

        del flownet
        torch.cuda.empty_cache()
        flownet = [torch.load(trt_engine_path) for _ in range(num_streams)]

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()

    if sc_threshold is not None:
        clip = sc_detect(clip, sc_threshold)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        remainder = n * factor_den % factor_num

        if remainder == 0 or (sc and f[0].props.get("_SceneChangeNext")):
            return f[0]

        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img0 = frame_to_tensor(f[0], device)
            img1 = frame_to_tensor(f[1], device)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            timestep = torch.full((1, 1, img0.shape[2], img0.shape[3]), remainder / factor_num, device=device)
            timestep = timestep.to(memory_format=torch.channels_last)

            if trt:
                output = flownet[local_index](img0, img1, timestep)
            else:
                output = flownet(img0, img1, timestep)

            return tensor_to_frame(output[:, :, :h, :w], f[0].copy())

    clip0 = vs.core.std.Interleave([clip] * factor_num)
    if factor_den > 1:
        clip0 = clip0.std.SelectEvery(cycle=factor_den, offsets=0)

    clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    clip1 = vs.core.std.Interleave([clip1] * factor_num)
    if factor_den > 1:
        clip1 = clip1.std.SelectEvery(cycle=factor_den, offsets=0)

    return clip0.std.FrameEval(lambda n: clip0.std.ModifyFrame([clip0, clip1], inference), clip_src=[clip0, clip1])


def sc_detect(clip: vs.VideoNode, threshold: float) -> vs.VideoNode:
    def copy_property(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props["_SceneChangePrev"] = f[1].props["_SceneChangePrev"]
        fout.props["_SceneChangeNext"] = f[1].props["_SceneChangeNext"]
        return fout

    sc_clip = clip.resize.Bicubic(format=vs.GRAY8, matrix_s="709").misc.SCDetect(threshold)
    return clip.std.FrameEval(lambda n: clip.std.ModifyFrame([clip, sc_clip], copy_property), clip_src=[clip, sc_clip])


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame
