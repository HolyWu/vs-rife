from __future__ import annotations

import math
import os
import sys
import warnings
from fractions import Fraction
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vapoursynth as vs
from torch._decomp import get_decompositions

__version__ = "5.4.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "Both operands of the binary elementwise op")
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
    "4.17.lite",
    "4.18",
    "4.19",
    "4.20",
    "4.21",
    "4.22",
    "4.22.lite",
    "4.23",
    "4.24",
    "4.25",
    "4.25.lite",
    "4.26",
]


@torch.inference_mode()
def rife(
    clip: vs.VideoNode,
    device_index: int = 0,
    model: str = "4.18",
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = True,
    sc_threshold: float | None = None,
    trt: bool = False,
    trt_static_shape: bool = True,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [1920, 1080],
    trt_max_shape: list[int] = [1920, 1080],
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
                                    Not supported in '4.0' and '4.1' models.
    :param trt_static_shape:        Build with static or dynamic shapes.
    :param trt_min_shape:           Min size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_opt_shape:           Opt size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_max_shape:           Max size of dynamic shapes. Ignored if trt_static_shape=True.
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

    if model not in models:
        raise vs.Error(f"rife: model must be one of {models}")

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

    if not trt_static_shape:
        raise vs.Error("rife: dynamic shapes is not allowed at the moment due to issues in Torch-TensorRT")

        if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
            raise vs.Error("rife: trt_min_shape must be a list with 2 items")

        if any(trt_min_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_min_shape must be at least 1")

        if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
            raise vs.Error("rife: trt_opt_shape must be a list with 2 items")

        if any(trt_opt_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_opt_shape must be at least 1")

        if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
            raise vs.Error("rife: trt_max_shape must be a list with 2 items")

        if any(trt_max_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_max_shape must be at least 1")

        if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
            raise vs.Error("rife: trt_min_shape must be less than trt_max_shape")

    if os.path.getsize(os.path.join(model_dir, "flownet_v4.0.pkl")) == 0:
        raise vs.Error("rife: model files have not been downloaded. run 'python -m vsrife' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    modulo = 32

    match model:
        case "4.0":
            from .IFNet_HDv3_v4_0 import IFNet

            Head = None
        case "4.1":
            from .IFNet_HDv3_v4_1 import IFNet

            Head = None
        case "4.2":
            from .IFNet_HDv3_v4_2 import IFNet

            Head = None
        case "4.3":
            from .IFNet_HDv3_v4_3 import IFNet

            Head = None
        case "4.4":
            from .IFNet_HDv3_v4_4 import IFNet

            Head = None
        case "4.5":
            from .IFNet_HDv3_v4_5 import IFNet

            Head = None
        case "4.6":
            from .IFNet_HDv3_v4_6 import IFNet

            Head = None
        case "4.7":
            from .IFNet_HDv3_v4_7 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.8":
            from .IFNet_HDv3_v4_8 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.9":
            from .IFNet_HDv3_v4_9 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.10":
            from .IFNet_HDv3_v4_10 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.11":
            from .IFNet_HDv3_v4_11 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.12":
            from .IFNet_HDv3_v4_12 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.12.lite":
            from .IFNet_HDv3_v4_12_lite import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 4, 4, 2, 1),
                )
            encode_channel = 4
        case "4.13":
            from .IFNet_HDv3_v4_13 import Head, IFNet
        case "4.13.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 4, 4, 2, 1),
                )
            encode_channel = 4
        case "4.14":
            from .IFNet_HDv3_v4_14 import Head, IFNet

            encode_channel = 8
        case "4.14.lite":
            from .IFNet_HDv3_v4_14_lite import Head, IFNet

            encode_channel = 8
        case "4.15":
            from .IFNet_HDv3_v4_15 import Head, IFNet

            encode_channel = 8
        case "4.15.lite":
            from .IFNet_HDv3_v4_15_lite import Head, IFNet

            encode_channel = 4
        case "4.16.lite":
            from .IFNet_HDv3_v4_16_lite import Head, IFNet

            encode_channel = 4
        case "4.17":
            from .IFNet_HDv3_v4_17 import Head, IFNet

            encode_channel = 8
        case "4.17.lite":
            from .IFNet_HDv3_v4_17_lite import Head, IFNet

            encode_channel = 4
        case "4.18":
            from .IFNet_HDv3_v4_18 import Head, IFNet

            encode_channel = 8
        case "4.19":
            from .IFNet_HDv3_v4_19 import Head, IFNet

            encode_channel = 8
        case "4.20":
            from .IFNet_HDv3_v4_20 import Head, IFNet

            encode_channel = 8
        case "4.21":
            from .IFNet_HDv3_v4_21 import Head, IFNet

            encode_channel = 8
        case "4.22":
            from .IFNet_HDv3_v4_22 import Head, IFNet

            encode_channel = 8
        case "4.22.lite":
            from .IFNet_HDv3_v4_22_lite import Head, IFNet

            encode_channel = 4
        case "4.23":
            from .IFNet_HDv3_v4_23 import Head, IFNet

            encode_channel = 8
        case "4.24":
            from .IFNet_HDv3_v4_24 import Head, IFNet

            encode_channel = 8
        case "4.25":
            from .IFNet_HDv3_v4_25 import Head, IFNet

            encode_channel = 4
            modulo = 64
        case "4.25.lite":
            from .IFNet_HDv3_v4_25_lite import Head, IFNet

            encode_channel = 4
            modulo = 128
        case "4.26":
            from .IFNet_HDv3_v4_26 import Head, IFNet

            encode_channel = 4
            modulo = 64

    model_name = f"flownet_v{model}.pkl"

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()

    w = clip.width
    h = clip.height
    tmp = max(modulo, int(modulo / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp
    padding = (0, pw - w, 0, ph - h)
    need_pad = any(p > 0 for p in padding)

    if sc_threshold is not None:
        clip = sc_detect(clip, sc_threshold)

    if trt:
        import tensorrt
        import torch_tensorrt

        if trt_static_shape:
            dimensions = f"{pw}x{ph}"
        else:
            for i in range(2):
                trt_min_shape[i] = math.ceil(trt_min_shape[i] / tmp) * tmp
                trt_opt_shape[i] = math.ceil(trt_opt_shape[i] / tmp) * tmp
                trt_max_shape[i] = math.ceil(trt_max_shape[i] / tmp) * tmp

            dimensions = (
                f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
                f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
                f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
            )

        flownet_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{dimensions}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_scale-{scale}"
                + f"_ensemble-{ensemble}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        encode_engine_path = flownet_engine_path + ".encode"

        if not os.path.isfile(flownet_engine_path) or (Head is not None and not os.path.isfile(encode_engine_path)):
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")

            flownet, encode = init_module(model_name, IFNet, scale, ensemble, device, dtype, Head)

            if encode is not None:
                flownet_example_inputs = (
                    torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                    torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                    torch.zeros([1, 1, ph, pw], dtype=dtype, device=device),
                    torch.zeros([2], dtype=torch.float, device=device),
                    torch.zeros([1, 2, ph, pw], dtype=torch.float, device=device),
                    torch.zeros([1, encode_channel, ph, pw], dtype=dtype, device=device),
                    torch.zeros([1, encode_channel, ph, pw], dtype=dtype, device=device),
                )

                encode_example_inputs = (torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),)
            else:
                flownet_example_inputs = (
                    torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                    torch.zeros([1, 3, ph, pw], dtype=dtype, device=device),
                    torch.zeros([1, 1, ph, pw], dtype=dtype, device=device),
                    torch.zeros([2], dtype=torch.float, device=device),
                    torch.zeros([1, 2, ph, pw], dtype=torch.float, device=device),
                )

            if trt_static_shape:
                flownet_dynamic_shapes = None
                encode_dynamic_shapes = None

                flownet_inputs = flownet_example_inputs

                if encode is not None:
                    encode_inputs = encode_example_inputs
            else:
                trt_min_shape.reverse()
                trt_opt_shape.reverse()
                trt_max_shape.reverse()

                _height = torch.export.Dim("height", min=trt_min_shape[0] // tmp, max=trt_max_shape[0] // tmp)
                _width = torch.export.Dim("width", min=trt_min_shape[1] // tmp, max=trt_max_shape[1] // tmp)
                dim_height = _height * tmp
                dim_width = _width * tmp

                if encode is not None:
                    flownet_dynamic_shapes = {
                        "img0": {2: dim_height, 3: dim_width},
                        "img1": {2: dim_height, 3: dim_width},
                        "timestep": {2: dim_height, 3: dim_width},
                        "tenFlow_div": {},
                        "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                        "f0": {2: dim_height, 3: dim_width},
                        "f1": {2: dim_height, 3: dim_width},
                    }

                    encode_dynamic_shapes = ({2: dim_height, 3: dim_width},)

                    flownet_inputs = [
                        torch_tensorrt.Input(
                            min_shape=[1, 3] + trt_min_shape,
                            opt_shape=[1, 3] + trt_opt_shape,
                            max_shape=[1, 3] + trt_max_shape,
                            dtype=dtype,
                            name="img0",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 3] + trt_min_shape,
                            opt_shape=[1, 3] + trt_opt_shape,
                            max_shape=[1, 3] + trt_max_shape,
                            dtype=dtype,
                            name="img1",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 1] + trt_min_shape,
                            opt_shape=[1, 1] + trt_opt_shape,
                            max_shape=[1, 1] + trt_max_shape,
                            dtype=dtype,
                            name="timestep",
                        ),
                        torch_tensorrt.Input(
                            shape=[2],
                            dtype=torch.float,
                            name="tenFlow_div",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 2] + trt_min_shape,
                            opt_shape=[1, 2] + trt_opt_shape,
                            max_shape=[1, 2] + trt_max_shape,
                            dtype=torch.float,
                            name="backwarp_tenGrid",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, encode_channel] + trt_min_shape,
                            opt_shape=[1, encode_channel] + trt_opt_shape,
                            max_shape=[1, encode_channel] + trt_max_shape,
                            dtype=dtype,
                            name="f0",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, encode_channel] + trt_min_shape,
                            opt_shape=[1, encode_channel] + trt_opt_shape,
                            max_shape=[1, encode_channel] + trt_max_shape,
                            dtype=dtype,
                            name="f1",
                        ),
                    ]

                    encode_inputs = [
                        torch_tensorrt.Input(
                            min_shape=[1, 3] + trt_min_shape,
                            opt_shape=[1, 3] + trt_opt_shape,
                            max_shape=[1, 3] + trt_max_shape,
                            dtype=dtype,
                        )
                    ]
                else:
                    flownet_dynamic_shapes = {
                        "img0": {2: dim_height, 3: dim_width},
                        "img1": {2: dim_height, 3: dim_width},
                        "timestep": {2: dim_height, 3: dim_width},
                        "tenFlow_div": {},
                        "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                    }

                    flownet_inputs = [
                        torch_tensorrt.Input(
                            min_shape=[1, 3] + trt_min_shape,
                            opt_shape=[1, 3] + trt_opt_shape,
                            max_shape=[1, 3] + trt_max_shape,
                            dtype=dtype,
                            name="img0",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 3] + trt_min_shape,
                            opt_shape=[1, 3] + trt_opt_shape,
                            max_shape=[1, 3] + trt_max_shape,
                            dtype=dtype,
                            name="img1",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 1] + trt_min_shape,
                            opt_shape=[1, 1] + trt_opt_shape,
                            max_shape=[1, 1] + trt_max_shape,
                            dtype=dtype,
                            name="timestep",
                        ),
                        torch_tensorrt.Input(
                            shape=[2],
                            dtype=torch.float,
                            name="tenFlow_div",
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 2] + trt_min_shape,
                            opt_shape=[1, 2] + trt_opt_shape,
                            max_shape=[1, 2] + trt_max_shape,
                            dtype=torch.float,
                            name="backwarp_tenGrid",
                        ),
                    ]

            flownet_program = torch.export.export(
                flownet, flownet_example_inputs, dynamic_shapes=flownet_dynamic_shapes
            )
            flownet_program = flownet_program.run_decompositions(get_decompositions([torch.ops.aten.grid_sampler_2d]))

            flownet = torch_tensorrt.dynamo.compile(
                flownet_program,
                flownet_inputs,
                device=device,
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                use_explicit_typing=True,
            )

            torch_tensorrt.save(
                flownet, flownet_engine_path, output_format="torchscript", inputs=flownet_example_inputs
            )

            if encode is not None:
                encode_program = torch.export.export(
                    encode, encode_example_inputs, dynamic_shapes=encode_dynamic_shapes
                )

                encode = torch_tensorrt.dynamo.compile(
                    encode_program,
                    encode_inputs,
                    device=device,
                    enabled_precisions={dtype},
                    debug=trt_debug,
                    num_avg_timing_iters=4,
                    workspace_size=trt_workspace_size,
                    min_block_size=1,
                    max_aux_streams=trt_max_aux_streams,
                    optimization_level=trt_optimization_level,
                )

                torch_tensorrt.save(
                    encode, encode_engine_path, output_format="torchscript", inputs=encode_example_inputs
                )

        flownet = torch.jit.load(flownet_engine_path).eval()
        if Head is not None:
            encode = torch.jit.load(encode_engine_path).eval()
    else:
        flownet, encode = init_module(model_name, IFNet, scale, ensemble, device, dtype, Head)

    inf_stream = torch.cuda.Stream(device)
    inf_f2t_stream = torch.cuda.Stream(device)
    inf_t2f_stream = torch.cuda.Stream(device)

    inf_stream_lock = Lock()
    inf_f2t_stream_lock = Lock()
    inf_t2f_stream_lock = Lock()

    pinned_tensor = torch.empty([2, 3, clip.height, clip.width], dtype=dtype, pin_memory=True)

    if Head is not None:
        enc_stream = torch.cuda.Stream(device)
        enc_f2t_stream = torch.cuda.Stream(device)

        enc_stream_lock = Lock()
        enc_f2t_stream_lock = Lock()

    timestep = {}
    for i in range(1, factor_num):
        t = i * factor_den % factor_num / factor_num
        timestep[t] = torch.full([1, 1, ph, pw], t, dtype=dtype, device=device)

    tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)

    tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
    tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
    tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
    tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
    backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    torch.cuda.current_stream(device).synchronize()

    frame_cache = {}
    encode_cache = {}

    @torch.inference_mode()
    def encoding(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        with enc_f2t_stream_lock, torch.cuda.stream(enc_f2t_stream):
            img = frame_to_tensor(f, pinned_tensor[0], device)

            if need_pad:
                img = F.pad(img, padding)

            enc_f2t_stream.synchronize()

            frame_cache[n] = img

        with enc_stream_lock, torch.cuda.stream(enc_stream):
            output = encode(img)

            enc_stream.synchronize()

            encode_cache[n] = output

            return f

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with inf_f2t_stream_lock, torch.cuda.stream(inf_f2t_stream):
            if Head is not None:
                real_n = n * factor_den // factor_num
                real_n_next = min(real_n + 1, clip.num_frames - 1)

                cache_to_delete = real_n - 10

                if cache_to_delete >= 0:
                    if cache_to_delete in frame_cache:
                        del frame_cache[cache_to_delete]

                    if cache_to_delete in encode_cache:
                        del encode_cache[cache_to_delete]

            t = n * factor_den % factor_num / factor_num

            if t == 0 or (sc and f[0].props.get("_SceneChangeNext")):
                return f[0]

            if Head is not None:
                if real_n in frame_cache:
                    img0 = frame_cache[real_n]
                else:
                    img0 = frame_to_tensor(f[0], pinned_tensor[0], device)

                    if need_pad:
                        img0 = F.pad(img0, padding)

                if real_n_next in frame_cache:
                    img1 = frame_cache[real_n_next]
                else:
                    img1 = frame_to_tensor(f[1], pinned_tensor[1], device)

                    if need_pad:
                        img1 = F.pad(img1, padding)

                if real_n in encode_cache:
                    f0 = encode_cache[real_n]
                else:
                    f0 = encode(img0)

                if real_n_next in encode_cache:
                    f1 = encode_cache[real_n_next]
                else:
                    f1 = encode(img1)
            else:
                img0 = frame_to_tensor(f[0], pinned_tensor[0], device)
                img1 = frame_to_tensor(f[1], pinned_tensor[1], device)

                if need_pad:
                    img0 = F.pad(img0, padding)
                    img1 = F.pad(img1, padding)

            inf_f2t_stream.synchronize()

        with inf_stream_lock, torch.cuda.stream(inf_stream):
            if Head is not None:
                output = flownet(img0, img1, timestep[t], tenFlow_div, backwarp_tenGrid, f0, f1)
            else:
                output = flownet(img0, img1, timestep[t], tenFlow_div, backwarp_tenGrid)

            inf_stream.synchronize()

        with inf_t2f_stream_lock, torch.cuda.stream(inf_t2f_stream):
            if need_pad:
                output = output[:, :, :h, :w]

            return tensor_to_frame(output, f[0].copy(), inf_t2f_stream)

    if Head is not None:
        encoded = clip.std.FrameEval(lambda n: clip.std.ModifyFrame(clip, encoding), clip_src=clip)
    else:
        encoded = clip

    clip0 = vs.core.std.Interleave([encoded] * factor_num)
    clip1 = encoded.std.DuplicateFrames(encoded.num_frames - 1)[1:]
    clip1 = vs.core.std.Interleave([clip1] * factor_num)
    if factor_den > 1:
        clip0 = clip0[::factor_den]
        clip1 = clip1[::factor_den]

    return clip0.std.FrameEval(lambda n: clip0.std.ModifyFrame([clip0, clip1], inference), clip_src=[clip0, clip1])


def init_module(
    model_name: str,
    IFNet: nn.Module,
    scale: float,
    ensemble: bool,
    device: torch.device,
    dtype: torch.dtype,
    Head: nn.Module | nn.Sequential | None,
) -> tuple[nn.Module, nn.Module | None]:
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu", weights_only=True, mmap=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    with torch.device("meta"):
        flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval().to(device, dtype)

    if Head is not None:
        encode_state_dict = {k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k}

        if isinstance(Head, nn.Sequential):
            encode = Head
        else:
            with torch.device("meta"):
                encode = Head()
        encode.load_state_dict(encode_state_dict, assign=True)
        encode.eval().to(device, dtype)

        return flownet, encode

    return flownet, None


def sc_detect(clip: vs.VideoNode, threshold: float) -> vs.VideoNode:
    def copy_property(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props["_SceneChangePrev"] = f[1].props["_SceneChangePrev"]
        fout.props["_SceneChangeNext"] = f[1].props["_SceneChangeNext"]
        return fout

    sc_clip = clip.resize.Bicubic(format=vs.GRAY8, matrix_s="709").misc.SCDetect(threshold)
    return clip.std.FrameEval(lambda n: clip.std.ModifyFrame([clip, sc_clip], copy_property), clip_src=[clip, sc_clip])


def frame_to_tensor(frame: vs.VideoFrame, pinned_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            pinned_tensor[plane].copy_(torch.from_numpy(np.asarray(frame[plane]))).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    ).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.squeeze(0).detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame
