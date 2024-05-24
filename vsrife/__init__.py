from __future__ import annotations

import math
import os
import warnings
from fractions import Fraction
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

__version__ = "5.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "At pre-dispatch tracing")
warnings.filterwarnings("ignore", "Attempted to insert a get_attr Node with no underlying reference")
warnings.filterwarnings("ignore", "Node _run_on_acc_0_engine target _run_on_acc_0_engine _run_on_acc_0_engine of")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

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
    "4.14",
    "4.14.lite",
    "4.15",
    "4.15.lite",
    "4.16.lite",
    "4.17",
]

models_str = ""
for model in models:
    models_str += "'" + model + "', "
models_str = models_str[:-2]


@torch.inference_mode()
def rife(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 2,
    model: str = "4.15.lite",
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = True,
    sc_threshold: float | None = None,
    trt: bool = False,
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param model:                   Model to use.
    :param factor_num:              Numerator of factor for target frame rate.
    :param factor_den:              Denominator of factor for target frame rate.
                                    For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param fps_num:                 Numerator of target frame rate.
    :param fps_den:                 Denominator of target frame rate.
                                    Override `factor_num` and `factor_den` if specified.
    :param scale:                   Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                                    Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:                Smooth predictions in areas where the estimation is uncertain.
    :param sc:                      Avoid interpolating frames over scene changes.
    :param sc_threshold:            Threshold for scene change detection. Must be between 0.0 and 1.0.
                                    Leave the argument as None if the frames already have _SceneChangeNext property set.
    :param trt:                     Use TensorRT for high-performance inference.
                                    Not supported for '4.0' and '4.1' models.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
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
    dtype = torch.half if fp16 else torch.float

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
            from .IFNet_HDv3_v4_14 import IFNet
        case "4.14.lite":
            from .IFNet_HDv3_v4_14_lite import IFNet
        case "4.15":
            from .IFNet_HDv3_v4_15 import IFNet
        case "4.15.lite":
            from .IFNet_HDv3_v4_15_lite import IFNet
        case "4.16.lite":
            from .IFNet_HDv3_v4_16_lite import IFNet
        case "4.17":
            from .IFNet_HDv3_v4_17 import IFNet

    model_name = f"flownet_v{model}.pkl"

    state_dict = torch.load(os.path.join(model_dir, model_name), map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False)
    flownet.eval().to(device)
    if fp16:
        flownet.half()

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()

    w = clip.width
    h = clip.height
    tmp = max(128, int(128 / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp
    padding = (0, pw - w, 0, ph - h)

    if sc_threshold is not None:
        clip = sc_detect(clip, sc_threshold)

    if trt:
        import tensorrt
        import torch_tensorrt

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{pw}x{ph}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_scale-{scale}"
                + f"_ensemble-{ensemble}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ep"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            inputs = [
                torch.zeros((1, 3, ph, pw), dtype=dtype, device=device),
                torch.zeros((1, 3, ph, pw), dtype=dtype, device=device),
                torch.zeros((1, 1, ph, pw), dtype=dtype, device=device),
            ]

            flownet = torch_tensorrt.compile(
                flownet,
                ir="dynamo",
                inputs=inputs,
                enabled_precisions={dtype},
                debug=trt_debug,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                device=device,
            )

            torch_tensorrt.save(flownet, trt_engine_path, inputs=inputs)

        flownet = [torch.export.load(trt_engine_path).module() for _ in range(num_streams)]

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

            timestep = torch.full((1, 1, ph, pw), remainder / factor_num, dtype=dtype, device=device)

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
    return (
        torch.stack([torch.from_numpy(np.asarray(frame[plane])).to(device) for plane in range(frame.format.num_planes)])
        .unsqueeze(0)
        .clamp(0.0, 1.0)
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane])
    return frame
