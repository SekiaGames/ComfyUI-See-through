import os
import sys
import random
import uuid
from datetime import datetime

print("[SeeThrough] nodes.py: starting imports...", flush=True)

import torch
import numpy as np

import folder_paths
import comfy.model_management as mm
import comfy.utils


def _log_vram(label):
    """Log current GPU VRAM usage for profiling."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[SeeThrough VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

print("[SeeThrough] nodes.py: comfy imports OK", flush=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEETHROUGH_ROOT_DIR = os.path.join(CURRENT_DIR, "see-through")
SEETHROUGH_COMMON_DIR = os.path.join(SEETHROUGH_ROOT_DIR, "common")

print(f"[SeeThrough] CURRENT_DIR = {CURRENT_DIR}", flush=True)
print(f"[SeeThrough] SEETHROUGH_COMMON_DIR = {SEETHROUGH_COMMON_DIR}", flush=True)
print(f"[SeeThrough] common dir exists = {os.path.isdir(SEETHROUGH_COMMON_DIR)}", flush=True)

# Mock pycocotools if not installed (only used for mask RLE, not needed here)
try:
    import pycocotools  # noqa: F401
    print("[SeeThrough] pycocotools found", flush=True)
except ImportError:
    print("[SeeThrough] pycocotools not found, installing mock...", flush=True)
    import types as _types
    _mock_pycocotools = _types.ModuleType("pycocotools")
    _mock_mask = _types.ModuleType("pycocotools.mask")
    _mock_pycocotools.mask = _mock_mask
    sys.modules["pycocotools"] = _mock_pycocotools
    sys.modules["pycocotools.mask"] = _mock_mask

if SEETHROUGH_COMMON_DIR not in sys.path:
    sys.path.insert(0, SEETHROUGH_COMMON_DIR)
    print(f"[SeeThrough] Added to sys.path: {SEETHROUGH_COMMON_DIR}", flush=True)

if SEETHROUGH_ROOT_DIR not in sys.path:
    sys.path.insert(1, SEETHROUGH_ROOT_DIR)
    print(f"[SeeThrough] Added to sys.path: {SEETHROUGH_ROOT_DIR}", flush=True)

_st_conflict_backup = {}
for _prefix in ("utils", "modules"):
    for _key in list(sys.modules.keys()):
        if _key == _prefix or _key.startswith(_prefix + "."):
            _st_conflict_backup[_key] = sys.modules.pop(_key)
if _st_conflict_backup:
    print(f"[SeeThrough] Temporarily removed {len(_st_conflict_backup)} conflicting sys.modules entries: "
          f"{list(_st_conflict_backup.keys())[:10]}{'...' if len(_st_conflict_backup) > 10 else ''}", flush=True)

print("[SeeThrough] Importing see-through modules...", flush=True)
import cv2
from safetensors.torch import load_file

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, img_alpha_blending, smart_resize
from utils.torchcv import cluster_inpaint_part

print("[SeeThrough] All see-through imports OK", flush=True)

for _key, _mod in _st_conflict_backup.items():
    if _key not in sys.modules:
        sys.modules[_key] = _mod
del _st_conflict_backup

DEFAULT_LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
DEFAULT_LAYERDIFF_NF4_REPO = "24yearsold/seethroughv0.0.2_layerdiff3d_nf4"
DEFAULT_DEPTH_REPO = "layerdifforg/seethroughv0.0.1_marigold"
DEFAULT_DEPTH_NF4_REPO = "24yearsold/seethroughv0.0.1_marigold_nf4"

QUANT_MODES = ["none", "nf4"]

_NF4_REPO_MAP = {
    DEFAULT_LAYERDIFF_REPO: DEFAULT_LAYERDIFF_NF4_REPO,
    DEFAULT_DEPTH_REPO: DEFAULT_DEPTH_NF4_REPO,
}

VALID_BODY_PARTS_V2 = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
    "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

# 面部细节：对 2D 骨骼动画没有单独动的价值（脸永远是一整块），所以默认并回 face。
# 名字要按「去左右后缀」的基名匹配——v3 直接给 eyewhite-l / ears-r，v2 的眼睛
# 拆完叫 eyel / eyer / browl / browr，两套都得认。
# 顺序 = 叠放顺序（先列的垫在下面）：耳朵在脸皮后面，其余都在脸皮上面；
# 眼镜/耳饰压在最上层（眼镜盖住眼睛，耳饰在耳朵上面）。
FACE_DETAIL_BELOW = ("ears", "earl", "earr")
FACE_DETAIL_ABOVE = ("eyes", "eyel", "eyer", "eyewhite", "irides", "eyelash",
                     "eyebrow", "browl", "browr", "nose", "mouth",
                     "eyewear", "earwear")

folder_paths.add_model_folder_path("SeeThrough", os.path.join(folder_paths.models_dir, "SeeThrough"))


def _model_base_dirs():
    """All registered SeeThrough model dirs, extra_model_paths.yaml entries first when is_default."""
    return folder_paths.get_folder_paths("SeeThrough")


try:
    os.makedirs(_model_base_dirs()[0], exist_ok=True)
except OSError:
    pass


class SeeThrough_LayersData:
    """Output of GenerateLayers: raw RGBA layers + preprocessing info."""
    def __init__(self, layer_dict, fullpage, input_img, resolution, pad_size, pad_pos):
        self.layer_dict = layer_dict      # tag -> RGBA numpy (resolution x resolution)
        self.fullpage = fullpage           # center-padded input (resolution x resolution, RGBA)
        self.input_img = input_img         # original input (RGBA)
        self.resolution = resolution
        self.pad_size = pad_size
        self.pad_pos = pad_pos
        self.scale = pad_size[0] / resolution


class SeeThrough_LayersDepthData:
    """Output of GenerateDepth: layers + per-tag depth maps."""
    def __init__(self, layer_dict, depth_dict, fullpage, resolution):
        self.layer_dict = layer_dict      # tag -> RGBA numpy
        self.depth_dict = depth_dict      # tag -> float32 depth [0,1]
        self.fullpage = fullpage
        self.resolution = resolution


def _cast_non_quantized_params(model, dtype):
    """Cast non-quantized float parameters to dtype, leaving bitsandbytes 4-bit params and integer tensors untouched."""
    try:
        import bitsandbytes as bnb
        bnb_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        bnb_types = ()
    for module in model.modules():
        if bnb_types and isinstance(module, bnb_types):
            continue
        for param in module.parameters(recurse=False):
            if param.data.is_floating_point():
                param.data = param.data.to(dtype=dtype)
        for buf in module.buffers(recurse=False):
            if buf.data.is_floating_point():
                buf.data = buf.data.to(dtype=dtype)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scan_model_dirs():
    """Recursively find diffusers model dirs (containing model_index.json) up to 2 levels deep
    across every registered SeeThrough folder.
    Returns relative paths from their base dir (e.g. 'foo' or 'org/foo'), deduplicated."""
    found = []
    for base in _model_base_dirs():
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, "model_index.json")) and "." not in found:
            found.append(".")
        for name in sorted(os.listdir(base)):
            d1 = os.path.join(base, name)
            if not os.path.isdir(d1):
                continue
            if os.path.isfile(os.path.join(d1, "model_index.json")):
                if name not in found:
                    found.append(name)
                continue
            for sub in sorted(os.listdir(d1)):
                d2 = os.path.join(d1, sub)
                rel = f"{name}/{sub}"
                if os.path.isdir(d2) and os.path.isfile(os.path.join(d2, "model_index.json")) and rel not in found:
                    found.append(rel)
    return found


def _resolve_model_path(model_name):
    bases = _model_base_dirs()
    if model_name == ".":
        for base in bases:
            if os.path.isfile(os.path.join(base, "model_index.json")):
                return base
        return bases[0]
    for base in bases:
        local = os.path.join(base, model_name)
        if os.path.isdir(local):
            return local
    # When model_name is an org/repo style ID (e.g. "layerdifforg/seethroughv0.0.1_marigold"),
    # check if just the repo part exists locally (git clone creates dirs without the org prefix).
    basename = model_name.split("/")[-1]
    if basename != model_name:
        for base in bases:
            local_basename = os.path.join(base, basename)
            if os.path.isdir(local_basename):
                return local_basename
    return model_name


def _label_lr_split(labels, stats, id1, id2):
    label1 = (labels == id1).astype(np.uint8) * 255
    label2 = (labels == id2).astype(np.uint8) * 255
    stats1, stats2 = stats[id1], stats[id2]
    x1 = stats[id1][0] + stats[id1][2] / 2
    x2 = stats[id2][0] + stats[id2][2] / 2
    if x2 < x1:
        return label2, label1, stats2, stats1
    return label1, label2, stats1, stats2


def _process_cuts(img, depth, src_xyxy, tgt_bbox, mask=None):
    tx1, ty1, tx2, ty2 = tgt_bbox[:4]
    tx2 += tx1
    ty2 += ty1
    img = img[ty1:ty2, tx1:tx2].copy()
    depth = depth[ty1:ty2, tx1:tx2]
    depth_median = 1.0
    if mask is not None:
        mask = (mask[ty1:ty2, tx1:tx2].copy() > 15).astype(np.uint8)
        ksize = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
        mask = cv2.dilate(mask, element)
        img[..., -1] *= mask
        depth = 1 - (1 - depth) * mask
        if np.any(mask):
            depth_median = float(np.median(depth[mask > 0]))
    fxyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1], tx2 + src_xyxy[0], ty2 + src_xyxy[1]]
    return img, depth, fxyxy, depth_median


def _part_lr_split(tag, part_info):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        part_info["mask"].astype(np.uint8) * 255, connectivity=8)
    tag2pinfo = {}
    if len(stats) > 2:
        stats = np.array(stats)
        stats_order = np.argsort(stats[..., -1])[::-1][1:]
        arml_mask, armr_mask, statsl, statsr = _label_lr_split(labels, stats, stats_order[0], stats_order[1])
        img, depth, xyxy, dm = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl, mask=arml_mask)
        tag2pinfo[f"{tag}-r"] = {"img": img, "xyxy": xyxy, "depth": depth, "depth_median": dm, "tag": f"{tag}-r"}
        img, depth, xyxy, dm = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr, mask=armr_mask)
        tag2pinfo[f"{tag}-l"] = {"img": img, "xyxy": xyxy, "depth": depth, "depth_median": dm, "tag": f"{tag}-l"}
    else:
        tag2pinfo[tag] = part_info
    return tag2pinfo


def _tag_lr_split(tag, tag2pinfo):
    if tag in tag2pinfo:
        tag2pinfo.update(_part_lr_split(tag, tag2pinfo.pop(tag)))


def _detail_base(tag):
    """eyewhite-l -> eyewhite；ears-r -> ears。v2 的 eyel / browr 原样返回。"""
    for suf in ("-l", "-r"):
        if tag.endswith(suf):
            return tag[:-2]
    return tag


def _as_bool(value, default=True):
    """吃 BOOLEAN 输入的各种写法。

    老工作流缺 widget 时会传 None；前端按钮留下的字符串（"Download PSD"）也可能
    落进 BOOLEAN 的位置，`bool()` 会把它当真，所以只认明确的真值写法。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _is_dust_layer(img):
    """这一层是模型「这张图没有这个部位」的占位输出，不是内容。

    实测形态（1024 画布）：主内容就是右下角一个 11×13 的实心小块，位置固定在
    ~(1010, 1009)，另外散着几十个点；整层也就 70 来个像素。导出后是一张
    1024² 的「空白图」（bbox 被两头的散点撑满，连裁剪都裁不掉）。

    判据故意做得保守，宁可漏删也不误删：
      · 整层像素 < 画布的 0.02%（1024² 上是 210 px）——真实的耳饰之类虽然小，
        也在这个量级之上；
      · 而且主块要么是「又小又贴右下角」的退化块，要么撑不起整层（散成一片碎点）。
    注意不能按「主块绝对像素数」判：真实的虹膜只有 45 px，比退化块还小。
    """
    mask = img[..., -1] > 10
    total = int(mask.sum())
    if total == 0:
        return True
    h, w = mask.shape[:2]
    if total >= 0.0002 * w * h:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return True
    main = 1 + int(stats[1:, 4].argmax())
    bx, by, bw, bh = stats[main, :4]
    if bw <= 24 and bh <= 24 and bx >= 0.9 * w and by >= 0.9 * h:
        return True
    return int(stats[main, 4]) / float(total) < 0.35


def _drop_empty_layers(layer_dict):
    """把所有「占位输出」的层删掉（就地），返回被删的 tag。"""
    dropped = [t for t, im in layer_dict.items() if _is_dust_layer(im)]
    for t in dropped:
        del layer_dict[t]
    return dropped


def _detail_has_content(img):
    """这一层是真的画了东西，还是模型没找到该部位时吐的散碎点。

    实测（1024 画布）「没找到」有三种形态：
      · 几十个点摊在几百像素的大框里（nose / mouth 常见，bbox 密度 0.02%）；
      · 缩在右下角一个 10×13 的实心小块里（包围盒退化，固定在 ~(1010, 1009)）——
        这种块密度和连通性都正常，只能靠「又小又贴右下角」认；
      · 零星散块（最大连通块撑不起整层的一半）。
    真层的 bbox 密度最低也有 0.3%，而且最大连通块占整层的大头。
    """
    mask = img[..., -1] > 10
    total = int(mask.sum())
    if total < 16:
        return False
    nz = cv2.findNonZero(mask.astype(np.uint8))
    if nz is None:
        return False
    bx, by, bw, bh = cv2.boundingRect(nz)
    h, w = mask.shape[:2]
    if bw <= 24 and bh <= 24 and bx >= 0.9 * w and by >= 0.9 * h:
        return False
    if total / float(max(bw * bh, 1)) < 0.002:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    main = int(stats[1:, 4].max()) if len(stats) > 1 else 0
    return main / float(total) >= 0.35


def _merge_face_details(layer_dict, fullpage=None):
    """把眼睛/耳朵/鼻子/嘴巴并进 face（就地改 layer_dict）。返回 (并进去的, 丢掉的)。

    在 GenerateLayers 里调：那时 layer_dict 是模型的原始输出，**还没拆左右**，
    各层都是整张画布，直接按 alpha 叠就行。放到 PostProcess 也能用，但要赶在
    裁剪/拆分之前，否则拿到的是小图、位置只在 xyxy 里，叠不上。

    内容判空的细节层（多半是 nose / mouth）直接丢掉：并进 face 只会把散点糊到
    脸上，那比少一层更糟。
    """
    if "face" not in layer_dict:
        return [], []

    # nose / mouth 的 RGB 模型给得不准，用原图补（只在有 alpha 的地方补）。
    # 必须在合并之前做，否则这几个 tag 已经没了。
    if fullpage is not None:
        for t in ("nose", "mouth"):
            im = layer_dict.get(t)
            if im is not None and im.shape[:2] == fullpage.shape[:2]:
                im = im.copy()
                m = im[..., 3] > 15
                im[..., :3][m] = fullpage[..., :3][m]
                layer_dict[t] = im

    below, above = [], []
    gone = []
    for tag in list(layer_dict.keys()):
        base = _detail_base(tag)
        if base in FACE_DETAIL_BELOW:
            bucket = below
        elif base in FACE_DETAIL_ABOVE:
            bucket = above
        else:
            continue
        if not _detail_has_content(layer_dict[tag]):
            gone.append(tag)
            del layer_dict[tag]
            continue
        bucket.append(tag)
    if not below and not above:
        return [], gone

    def ordered(group, canonical):
        """按 canonical 的顺序排（= 叠放顺序），同一基名的左右两片按名字稳定排。"""
        return [t for base in canonical for t in sorted(group) if _detail_base(t) == base]

    order = ordered(below, FACE_DETAIL_BELOW) + ["face"] + ordered(above, FACE_DETAIL_ABOVE)
    imlist = []
    for t in order:
        im = layer_dict[t].copy()
        # 和 GenerateDepth 里 compose_list 的合成分支同一个阈值：太淡的边不算内容
        im[..., -1][im[..., -1] < 15] = 0
        imlist.append(im)

    layer_dict["face"] = img_alpha_blending(imlist, premultiplied=False)
    for t in below + above:
        del layer_dict[t]
    return below + above, gone


def _compute_depth_median(part_dict):
    img = part_dict.pop("img")
    part_dict.pop("mask", None)
    depth = part_dict.pop("depth")
    mask = img[..., -1] > 10
    depth_median = float(np.median(depth[mask])) if np.any(mask) else 1.0
    nz = cv2.findNonZero(mask.astype(np.uint8))
    if nz is not None:
        xywh = cv2.boundingRect(nz)
        cx1, cy1 = int(xywh[0]), int(xywh[1])
        cx2, cy2 = cx1 + int(xywh[2]), cy1 + int(xywh[3])
        depth = depth[cy1:cy2, cx1:cx2]
        img = img[cy1:cy2, cx1:cx2]
        if "xyxy" in part_dict:
            ox, oy = part_dict["xyxy"][0], part_dict["xyxy"][1]
            part_dict["xyxy"] = [ox + cx1, oy + cy1, ox + cx2, oy + cy2]
        else:
            part_dict["xyxy"] = [cx1, cy1, cx2, cy2]
    depth = np.clip(depth, 0, 1) * 255
    depth = np.round(depth).astype(np.uint8)
    part_dict["depth_median"] = depth_median
    part_dict["img"] = img
    part_dict["depth"] = depth
    return part_dict


def _crop_head(img, xywh):
    x, y, w, h = xywh
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h
    if w < iw // 2:
        px = min(iw - x - w, x, w // 5)
        x1 = min(max(x - px, 0), iw)
        x2 = min(max(x + w + px, 0), iw)
    if h < ih // 2:
        py = min(ih - y - h, y, h // 5)
        y2 = min(max(y + h + py, 0), ih)
        y1 = min(max(y - py, 0), ih)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def _make_preview(tag2pinfo, resolution):
    drawables = list(tag2pinfo.values())
    if drawables:
        blended = img_alpha_blending(drawables, premultiplied=False, final_size=(resolution, resolution))
    else:
        blended = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    preview = blended[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(preview).unsqueeze(0)


class SeeThrough_LoadLayerDiffModel:
    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_LAYERDIFF_REPO, DEFAULT_LAYERDIFF_NF4_REPO]
        return {
            "required": {
                "model": (model_list, {"default": DEFAULT_LAYERDIFF_REPO,
                                       "tooltip": "HuggingFace repo ID or local model folder in models/SeeThrough/"}),
            },
            "optional": {
                "vae_ckpt": ("STRING", {"default": "",
                                        "tooltip": "Optional path to a custom VAE checkpoint (.safetensors)"}),
                "unet_ckpt": ("STRING", {"default": "",
                                         "tooltip": "Optional path to a custom UNet checkpoint"}),
                "quant_mode": (QUANT_MODES, {"default": "none",
                                              "tooltip": "Quantization mode: 'none' for bf16, 'nf4' for 4-bit NormalFloat quantization (~8GB VRAM). Requires bitsandbytes."}),
                "cache_tag_embeds": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Pre-compute and cache tag embeddings, then unload text encoders to save VRAM"}),
                "group_offload": ("BOOLEAN", {"default": False,
                                               "tooltip": "Enable group offload to reduce peak VRAM (~10GB) at cost of ~1.5x slower speed"}),
                "auto_download": ("BOOLEAN", {"default": False,
                                               "tooltip": "If the model is not found locally, download it from HuggingFace. Off by default: an accidental download is hundreds of MB to several GB, so make it an explicit choice."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERDIFF_MODEL",)
    RETURN_NAMES = ("layerdiff_model",)
    FUNCTION = "load_model"
    CATEGORY = "SeeThrough"

    def load_model(self, model, vae_ckpt="", unet_ckpt="", quant_mode="none", cache_tag_embeds=True, group_offload=False, auto_download=False):
        use_nf4 = quant_mode == "nf4"
        dtype = torch.bfloat16

        if use_nf4 and model in _NF4_REPO_MAP:
            model = _NF4_REPO_MAP[model]
            print(f"[SeeThrough] quant_mode=nf4: auto-switched to NF4 repo {model}", flush=True)

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)
        local_only = is_local or not auto_download

        print(f"[SeeThrough] Loading LayerDiff model from: {pretrained} "
              f"(quant_mode={quant_mode}, local={is_local}, local_files_only={local_only})", flush=True)
        trans_vae = TransparentVAE.from_pretrained(pretrained, subfolder="trans_vae", local_files_only=local_only)

        if unet_ckpt:
            print(f"[SeeThrough] Loading custom UNet from: {unet_ckpt}", flush=True)
            unet = UNetFrameConditionModel.from_pretrained(unet_ckpt)
        else:
            unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", local_files_only=local_only)

        # 0.5.0 的 KDiffusionStableDiffusionXLPipeline 在 scheduler=None 时会写死去拉
        # frankjoshua/juggernautXL_version6Rundiffusion 的 scheduler/scheduler_config.json，
        # 离线或镜像不通时必然报 “does not appear to have a file named scheduler_config.json”。
        # 本地模型目录自带同一份配置（DPMPP_2M_SDE + final_sigmas_type=zero，SDXL betas），
        # 所以优先从本地读，读不到才退回原行为。
        scheduler = None
        _sched_dir = os.path.join(pretrained, "scheduler")
        if os.path.isfile(os.path.join(_sched_dir, "scheduler_config.json")):
            from diffusers import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                pretrained, subfolder="scheduler", local_files_only=True)
            print("[SeeThrough] Scheduler from local model dir (no juggernaut fetch)", flush=True)

        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            pretrained, trans_vae=trans_vae, unet=unet, scheduler=scheduler, local_files_only=local_only)

        if vae_ckpt:
            print(f"[SeeThrough] Loading custom VAE from: {vae_ckpt}", flush=True)
            td_sd, vae_sd = {}, {}
            sd = load_file(vae_ckpt)
            for k, v in sd.items():
                if k.startswith("trans_decoder."):
                    td_sd[k[len("trans_decoder."):]] = v
                elif k.startswith("vae."):
                    vae_sd[k.replace("vae.", "")] = v
            if vae_sd:
                pipeline.vae.load_state_dict(vae_sd)
            if td_sd:
                pipeline.trans_vae.decoder.load_state_dict(td_sd)

        pipeline.vae.to(dtype=dtype)
        pipeline.trans_vae.to(dtype=dtype)

        if use_nf4:
            print("[SeeThrough] NF4 mode: casting non-quantized parameters to bf16", flush=True)
            _cast_non_quantized_params(pipeline.unet, dtype)
            _cast_non_quantized_params(pipeline.text_encoder, dtype)
            _cast_non_quantized_params(pipeline.text_encoder_2, dtype)
        else:
            pipeline.unet.to(dtype=dtype)
            pipeline.text_encoder.to(dtype=dtype)
            pipeline.text_encoder_2.to(dtype=dtype)

        pipeline._st_group_offload = False
        if group_offload:
            if hasattr(pipeline, 'enable_group_offload'):
                print("[SeeThrough] Enabling group offload for LayerDiff pipeline", flush=True)
                # onload_device 写死 'cuda' 在 macOS 上会直接炸（MPS 机器没有 cuda 设备），
                # 用当前实际设备，显存放不下的块退回 CPU。
                pipeline.enable_group_offload(mm.get_torch_device(), offload_device='cpu',
                                              num_blocks_per_group=1)
                pipeline._st_group_offload = True
            else:
                print("[SeeThrough] WARNING: group_offload requires diffusers >= 0.37.0, skipping. Please upgrade: pip install diffusers>=0.37.0", flush=True)

        if cache_tag_embeds:
            if not pipeline._st_group_offload:
                device = mm.get_torch_device()
                if not use_nf4:
                    pipeline.text_encoder.to(device)
                    pipeline.text_encoder_2.to(device)
            print("[SeeThrough] Caching tag embeddings and unloading text encoders...", flush=True)
            pipeline.cache_tag_embeds(unload_textencoders=True)
            _log_vram("After cache_tag_embeds (text encoders unloaded)")

        _log_vram("LayerDiff model loaded (CPU)")
        print(f"[SeeThrough] LayerDiff model loaded (quant_mode={quant_mode})", flush=True)
        return (pipeline,)

class SeeThrough_LoadDepthModel:
    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_DEPTH_REPO, DEFAULT_DEPTH_NF4_REPO]
        return {
            "required": {
                "model": (model_list, {"default": DEFAULT_DEPTH_REPO,
                                       "tooltip": "HuggingFace repo ID or local model folder in models/SeeThrough/"}),
            },
            "optional": {
                "quant_mode": (QUANT_MODES, {"default": "none",
                                              "tooltip": "Quantization mode: 'none' for bf16, 'nf4' for 4-bit NormalFloat quantization. Requires bitsandbytes."}),
                "cache_tag_embeds": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Pre-compute empty text embedding and unload text encoder to save VRAM"}),
                "group_offload": ("BOOLEAN", {"default": False,
                                               "tooltip": "Enable group offload to reduce peak VRAM at cost of slower speed"}),
                "auto_download": ("BOOLEAN", {"default": False,
                                               "tooltip": "If the model is not found locally, download it from HuggingFace. Off by default: an accidental download is hundreds of MB to several GB, so make it an explicit choice."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "load_model"
    CATEGORY = "SeeThrough"

    def load_model(self, model, quant_mode="none", cache_tag_embeds=True, group_offload=False, auto_download=False):
        use_nf4 = quant_mode == "nf4"
        dtype = torch.bfloat16

        if use_nf4 and model in _NF4_REPO_MAP:
            model = _NF4_REPO_MAP[model]
            print(f"[SeeThrough] quant_mode=nf4: auto-switched to NF4 repo {model}", flush=True)

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)
        local_only = is_local or not auto_download

        print(f"[SeeThrough] Loading Marigold depth model from: {pretrained} "
              f"(quant_mode={quant_mode}, local={is_local}, local_files_only={local_only})", flush=True)
        unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", local_files_only=local_only)
        pipeline = MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet, local_files_only=local_only)

        if use_nf4:
            print("[SeeThrough] NF4 mode: casting non-quantized parameters to bf16", flush=True)
            pipeline.vae.to(dtype=dtype)
            _cast_non_quantized_params(pipeline.unet, dtype)
            _cast_non_quantized_params(pipeline.text_encoder, dtype)
        else:
            pipeline.to(dtype=dtype)

        pipeline._st_group_offload = False
        if group_offload:
            if hasattr(pipeline, 'enable_group_offload'):
                print("[SeeThrough] Enabling group offload for Marigold pipeline", flush=True)
                # onload_device 写死 'cuda' 在 macOS 上会直接炸（MPS 机器没有 cuda 设备），
                # 用当前实际设备，显存放不下的块退回 CPU。
                pipeline.enable_group_offload(mm.get_torch_device(), offload_device='cpu',
                                              num_blocks_per_group=1)
                pipeline._st_group_offload = True
            else:
                print("[SeeThrough] WARNING: group_offload requires diffusers >= 0.37.0, skipping. Please upgrade: pip install diffusers>=0.37.0", flush=True)

        if cache_tag_embeds:
            if not pipeline._st_group_offload:
                device = mm.get_torch_device()
                if not use_nf4:
                    pipeline.text_encoder.to(device)
            print("[SeeThrough] Caching empty text embedding and unloading text encoder...", flush=True)
            pipeline.cache_tag_embeds(unload_textencoders=True)
            _log_vram("After Marigold cache_tag_embeds (text encoder unloaded)")

        _log_vram("Depth model loaded (CPU)")
        print(f"[SeeThrough] Depth model loaded (quant_mode={quant_mode})", flush=True)
        return (pipeline,)

class SeeThrough_GenerateLayers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "resolution": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
            # 这两个是后加的开关，放 optional：老工作流的 widgets_values 比节点定义短，
            # 放 required 会在执行时报「缺少必需的输入：drop_empty_layers」。
            "optional": {
                "merge_face_details": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep eyes / eyebrows / ears / nose / mouth and (when present) glasses / "
                               "earrings inside the face layer instead of exporting them separately. "
                               "A face is always one piece in a 2D rig, and nose / mouth often come out "
                               "as scattered dust. Turn off to get the raw split.",
                }),
                "drop_empty_layers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Delete layers the model only filled with a placeholder (a 11x13 dot in the "
                               "bottom-right corner, ~70 px in total). They export as blank 1024x1024 PNGs "
                               "and are useless downstream. Real small accessories survive: the check also "
                               "requires the layer to be under 0.02% of the canvas.",
                }),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERS", "IMAGE")
    RETURN_NAMES = ("layers", "preview")
    FUNCTION = "generate"
    CATEGORY = "SeeThrough"

    def generate(self, image, layerdiff_model, seed=42, resolution=1280, num_inference_steps=30,
                 merge_face_details=True, drop_empty_layers=True):
        # 显式传 None（老工作流没有这两个 widget）时按默认「开」处理
        merge_face_details = _as_bool(merge_face_details, True)
        drop_empty_layers = _as_bool(drop_empty_layers, True)
        pipeline = layerdiff_model
        device = mm.get_torch_device()
        offload = torch.device("cpu")
        seed_everything(seed)

        # Convert ComfyUI IMAGE to numpy RGBA
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 3:
            img_np = np.concatenate([img_np, np.full((*img_np.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        input_img = img_np.copy()

        fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
        scale = pad_size[0] / resolution

        tag_version = pipeline.unet.get_tag_version()
        layer_dict = {}

        print(f"[SeeThrough] GenerateLayers: tag_version={tag_version}, resolution={resolution}, steps={num_inference_steps}", flush=True)
        _log_vram("GenerateLayers start")

        is_group_offload = getattr(pipeline, '_st_group_offload', False)

        has_cached = len(pipeline._cached_prompt_embeds) > 0
        if has_cached:
            print("[SeeThrough] Using cached tag embeddings", flush=True)
            encode_fn = pipeline.encode_cropped_prompt_77tokens_cached
        else:
            if not is_group_offload:
                pipeline.text_encoder.to(device)
                pipeline.text_encoder_2.to(device)
                _log_vram("Text encoders loaded to GPU")
            encode_fn = pipeline.encode_cropped_prompt_77tokens

        if tag_version == "v2":
            prompt_embeds, pooled_prompt_embeds = encode_fn(VALID_BODY_PARTS_V2)
        elif tag_version == "v3":
            body_tags = ["front hair", "back hair", "head", "neck", "neckwear",
                         "topwear", "handwear", "bottomwear", "legwear", "footwear",
                         "tail", "wings", "objects"]
            head_tags = ["headwear", "face", "irides", "eyebrow", "eyewhite",
                         "eyelash", "eyewear", "ears", "earwear", "nose", "mouth"]
            body_embeds, body_pooled = encode_fn(body_tags)
            head_embeds, head_pooled = encode_fn(head_tags)
        else:
            raise ValueError(f"Unknown tag version: {tag_version}")

        if not has_cached and not is_group_offload:
            pipeline.text_encoder.to(offload)
            pipeline.text_encoder_2.to(offload)
            _log_vram("Text encoders offloaded to CPU")

        if not is_group_offload:
            pipeline.unet.to(device)
            pipeline.vae.to(device)
        pipeline.trans_vae.to(device)
        mm.soft_empty_cache()
        _log_vram("UNet+VAE on GPU, ready for diffusion")

        rng = torch.Generator(device=device).manual_seed(seed)

        print("[SeeThrough] Text encoded, text encoders offloaded to CPU", flush=True)

        if tag_version == "v2":
            out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                           generator=rng, guidance_scale=1.0,
                           prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                           fullpage=fullpage)
            _log_vram("v2 diffusion complete")
            for rst, tag in zip(out.images, VALID_BODY_PARTS_V2):
                layer_dict[tag] = rst

        elif tag_version == "v3":
            out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                           generator=rng, guidance_scale=1.0,
                           prompt_embeds=body_embeds, pooled_prompt_embeds=body_pooled,
                           fullpage=fullpage, group_index=0)
            _log_vram("v3 body diffusion complete")
            for rst, tag in zip(out.images, body_tags):
                layer_dict[tag] = rst

            head_img = out.images[2]
            nz = cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
            if nz is not None:
                hx0, hy0, hw, hh = cv2.boundingRect(nz)
                hx = int(hx0 * scale) - pad_pos[0]
                hy = int(hy0 * scale) - pad_pos[1]
                input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, int(hw * scale), int(hh * scale)])
                hx1 = int(hx1 / scale + pad_pos[0] / scale)
                hy1 = int(hy1 / scale + pad_pos[1] / scale)
                ih, iw = input_head.shape[:2]
                input_head, head_pad_size, head_pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

                out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                               generator=rng, guidance_scale=1.0,
                               prompt_embeds=head_embeds, pooled_prompt_embeds=head_pooled,
                               fullpage=input_head, group_index=1)
                _log_vram("v3 head diffusion complete")

                canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
                coords = np.array([head_pad_pos[1], head_pad_pos[1] + ih, head_pad_pos[0], head_pad_pos[0] + iw])
                py1, py2, px1, px2 = (coords / scale).astype(np.int64)
                scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))

                for rst, tag in zip(out.images, head_tags):
                    rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
                    full = canvas.copy()
                    full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
                    layer_dict[tag] = full

        if not is_group_offload:
            pipeline.unet.to(offload)
            pipeline.vae.to(offload)
        pipeline.trans_vae.to(offload)
        mm.soft_empty_cache()
        _log_vram("GenerateLayers offloaded to CPU")
        print(f"[SeeThrough] GenerateLayers complete: {len(layer_dict)} layers", flush=True)

        # 面部细节并回 face。放在这里而不是 PostProcess：模型原本就没给 face 画五官，
        # 五官永远单独一层；而做 2D 骨骼动画时脸是一整块。这里各层还没拆左右、
        # 都是整张画布，叠起来不用管 xyxy。
        # 注意：这不省时间/显存——depth 阶段的 batch 恒为 19 个槽位（见 GenerateDepth），
        # 合并只是让输出和下游（PSD / Spine 工作台）少十几层。
        if merge_face_details:
            merged, empty = _merge_face_details(layer_dict, fullpage)
            if merged:
                print(f"[SeeThrough] Face details merged into 'face': {', '.join(sorted(merged))}", flush=True)
            if empty:
                print(f"[SeeThrough] Face details dropped (no content): {', '.join(sorted(empty))}", flush=True)

        # 模型没找到的部位会留一个「占位小块」，导出就是一张空白 png。直接删掉。
        if drop_empty_layers:
            dust = _drop_empty_layers(layer_dict)
            if dust:
                print(f"[SeeThrough] Empty layers dropped: {', '.join(sorted(dust))}", flush=True)

        print(f"[SeeThrough] GenerateLayers final: {len(layer_dict)} layers", flush=True)

        layers_data = SeeThrough_LayersData(layer_dict, fullpage, input_img, resolution, pad_size, pad_pos)

        preview_dict = {}
        for tag, img in layer_dict.items():
            mask = img[..., -1] > 10
            if np.any(mask):
                preview_dict[tag] = {"img": img, "xyxy": [0, 0, resolution, resolution]}
        preview = _make_preview(preview_dict, resolution)

        return (layers_data, preview)


class SeeThrough_GenerateDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers": ("SEETHROUGH_LAYERS",),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "resolution_depth": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 64,
                                              "tooltip": "Resolution for depth inference. -1 uses the same resolution as layers. Lower values save VRAM and speed up inference."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERS_DEPTH", "IMAGE")
    RETURN_NAMES = ("layers_depth", "preview")
    FUNCTION = "generate"
    CATEGORY = "SeeThrough"

    def generate(self, layers, depth_model, seed=42, resolution_depth=-1):
        layer_dict = layers.layer_dict
        fullpage = layers.fullpage
        resolution = layers.resolution
        marigold = depth_model
        device = mm.get_torch_device()
        offload = torch.device("cpu")
        is_group_offload = getattr(marigold, '_st_group_offload', False)

        print("[SeeThrough] GenerateDepth: running Marigold...", flush=True)
        _log_vram("GenerateDepth start")

        empty_array = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        blended_alpha = np.zeros((resolution, resolution), dtype=np.float32)
        compose_list = {"eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
                        "hair": ["back hair", "front hair"]}

        img_list = []
        for tag in VALID_BODY_PARTS_V2:
            if tag in layer_dict:
                tag_arr = layer_dict[tag].copy()
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                img_list.append(tag_arr)
            else:
                img_list.append(empty_array.copy())

        compose_dict = {}
        for c, clist in compose_list.items():
            imlist, taglist = [], []
            for t in clist:
                if t in layer_dict:
                    tag_arr = layer_dict[t].copy()
                    tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                    imlist.append(tag_arr)
                    taglist.append(t)
            if imlist:
                composed = img_alpha_blending(imlist, premultiplied=False)
                img_list[VALID_BODY_PARTS_V2.index(c)] = composed
                compose_dict[c] = {"taglist": taglist, "imlist": imlist}

        for img in img_list:
            blended_alpha += img[..., -1].astype(np.float32) / 255
        blended_alpha = np.clip(blended_alpha, 0, 1) * 255
        blended_alpha = blended_alpha.astype(np.uint8)

        fullpage_for_depth = fullpage.copy()
        fullpage_for_depth[..., -1] = blended_alpha
        img_list.append(fullpage_for_depth)

        src_h, src_w = resolution, resolution
        if resolution_depth > 0 and resolution_depth != resolution:
            depth_res = resolution_depth
            depth_res = max(64, (depth_res // 8) * 8)
            img_list_input = [smart_resize(img, (depth_res, depth_res)) for img in img_list]
            print(f"[SeeThrough] Depth inference at resolution {depth_res} (layers at {resolution})", flush=True)
        else:
            img_list_input = img_list
            depth_res = resolution

        if not is_group_offload:
            marigold.unet.to(device)
            marigold.vae.to(device)
            mm.soft_empty_cache()
            _log_vram("Marigold on GPU")
            print("[SeeThrough] Marigold pipeline moved to GPU", flush=True)

        seed_everything(seed)
        pipe_out = marigold(color_map=None, show_progress_bar=False, img_list=img_list_input)
        _log_vram("Marigold inference complete")
        depth_pred = pipe_out.depth_tensor.to(device="cpu", dtype=torch.float32).numpy()

        if depth_res != resolution:
            depth_pred = np.stack([smart_resize(d, (src_h, src_w)) for d in depth_pred])

        if not is_group_offload:
            marigold.unet.to(offload)
            marigold.vae.to(offload)
            mm.soft_empty_cache()
            _log_vram("GenerateDepth offloaded to CPU")

        depth_dict = {}
        for ii, tag in enumerate(VALID_BODY_PARTS_V2):
            depth = depth_pred[ii]
            if tag in compose_dict:
                mask_accum = blended_alpha > 256  # all-False
                for t, im in zip(compose_dict[tag]["taglist"][::-1], compose_dict[tag]["imlist"][::-1]):
                    mask_local = im[..., -1] > 15
                    mask_invis = np.bitwise_and(mask_accum, mask_local)
                    depth_local = np.full((resolution, resolution), fill_value=1.0, dtype=np.float32)
                    depth_local[mask_local] = depth[mask_local]
                    if np.any(mask_invis):
                        vis = np.bitwise_and(mask_local, np.bitwise_not(mask_invis))
                        if np.any(vis):
                            depth_local[mask_invis] = np.median(depth[vis])
                    mask_accum = np.bitwise_or(mask_accum, mask_local)
                    depth_dict[t] = depth_local
            else:
                depth_dict[tag] = np.clip(depth, 0, 1).astype(np.float32)

        print(f"[SeeThrough] GenerateDepth complete: {len(depth_dict)} depth maps, Marigold offloaded to CPU", flush=True)

        result = SeeThrough_LayersDepthData(layer_dict, depth_dict, fullpage, resolution)

        # Preview: blend with depth info
        preview_dict = {}
        for tag in layer_dict:
            img = layer_dict[tag]
            if tag in depth_dict and np.any(img[..., -1] > 10):
                preview_dict[tag] = {"img": img, "depth": depth_dict[tag], "xyxy": [0, 0, resolution, resolution]}
        preview = _make_preview(preview_dict, resolution)

        return (result, preview)

class SeeThrough_PostProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers_depth": ("SEETHROUGH_LAYERS_DEPTH",),
                "tblr_split": ("BOOLEAN", {"default": True,
                                           "tooltip": "Split symmetric parts (eyes, ears, handwear) into left/right"}),
                "use_lama": ("BOOLEAN", {"default": True,
                                         "tooltip": "Use LaMa inpainting for hair splitting (better quality). Falls back to OpenCV if disabled."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_PARTS", "IMAGE")
    RETURN_NAMES = ("parts", "preview")
    FUNCTION = "process"
    CATEGORY = "SeeThrough"

    def process(self, layers_depth, tblr_split=True, use_lama=True):
        layer_dict = layers_depth.layer_dict
        depth_dict = layers_depth.depth_dict
        fullpage = layers_depth.fullpage
        resolution = layers_depth.resolution

        print("[SeeThrough] PostProcess: splitting & clustering...", flush=True)

        # Build tag2pinfo
        tag2pinfo = {}
        for tag in layer_dict:
            img = layer_dict[tag]
            if tag not in depth_dict:
                continue
            depth = depth_dict[tag]
            mask = img[..., -1] > 10
            if not np.any(mask):
                continue
            tag2pinfo[tag] = {"img": img, "depth": depth, "xyxy": [0, 0, resolution, resolution],
                              "mask": mask, "tag": tag}

        # Nose/mouth color restoration
        # 只有关掉 GenerateLayers 的 merge_face_details 时才会走到这里——那时候
        # nose/mouth 还是独立层，RGB 得从原图补回来。
        for restore_tag in ("nose", "mouth"):
            if restore_tag in tag2pinfo:
                pinfo = tag2pinfo[restore_tag]
                src_h, src_w = pinfo["img"].shape[:2]
                fp_h, fp_w = fullpage.shape[:2]
                if src_h == fp_h and src_w == fp_w:
                    pinfo["img"][..., :3] = fullpage[..., :3]
                else:
                    x1, y1 = pinfo["xyxy"][0], pinfo["xyxy"][1]
                    crop = fullpage[y1:min(y1 + src_h, fp_h), x1:min(x1 + src_w, fp_w), :3]
                    pinfo["img"][:crop.shape[0], :crop.shape[1], :3] = crop

        # Eye splitting (v2 composite 'eyes')
        if "eyes" in tag2pinfo:
            part_info = tag2pinfo.pop("eyes")
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                part_info["mask"].astype(np.uint8) * 255, connectivity=8)
            if len(stats) > 2:
                stats_arr = np.array(stats)
                if len(stats_arr[..., -1]) >= 5:
                    stats_order = np.argsort(stats_arr[..., -1])[::-1][1:]
                    eyel_mask, eyer_mask, statsl, statsr = _label_lr_split(labels, stats_arr, stats_order[0], stats_order[1])
                    img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl)
                    tag2pinfo["eyer"] = {"img": img, "xyxy": xyxy, "depth": depth}
                    img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr)
                    tag2pinfo["eyel"] = {"img": img, "xyxy": xyxy, "depth": depth}
                    if len(stats_order) >= 4:
                        browl_mask, browr_mask, statsl, statsr = _label_lr_split(labels, stats_arr, stats_order[2], stats_order[3])
                        img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl)
                        tag2pinfo["browr"] = {"img": img, "xyxy": xyxy, "depth": depth}
                        img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr)
                        tag2pinfo["browl"] = {"img": img, "xyxy": xyxy, "depth": depth}
                else:
                    tag2pinfo["eyes"] = part_info
            else:
                tag2pinfo["eyes"] = part_info

        # Left-right splitting
        if tblr_split:
            _tag_lr_split("handwear", tag2pinfo)
            for eye_tag in ["eyewhite", "irides", "eyelash", "eyebrow"]:
                _tag_lr_split(eye_tag, tag2pinfo)
            _tag_lr_split("ears", tag2pinfo)
            # 腿和鞋也试着分左右：只有「左腿和右腿是两个不连通的块」时才拆得开。
            # 侧面/3-4 视角两条腿在画面上叠在一起，是一个连通块，这里会原样保留整块——
            # 要拆开得换正面（或两腿分开）的站姿图，或者走按关节切件的路线。
            for limb_tag in ["legwear", "footwear"]:
                _tag_lr_split(limb_tag, tag2pinfo)

            if "hair" in tag2pinfo:
                part_info = tag2pinfo.pop("hair")
                try:
                    inpaint_mode = "lama" if use_lama else "cv2"
                    parts = cluster_inpaint_part(inpaint=inpaint_mode, **part_info)
                    parts.sort(key=lambda x: x["depth_median"])
                    tag2pinfo["hairf"] = parts[0]
                    tag2pinfo["hairb"] = parts[1]
                except Exception as e:
                    print(f"[SeeThrough] Hair clustering failed: {e}, keeping as-is", flush=True)
                    tag2pinfo["hair"] = part_info

        # Crop + depth_median
        for tag in list(tag2pinfo.keys()):
            pinfo = tag2pinfo[tag]
            if "img" in pinfo and "depth" in pinfo:
                _compute_depth_median(pinfo)
            pinfo["tag"] = tag

        # Depth ordering adjustments
        if "face" in tag2pinfo:
            face_dm = tag2pinfo["face"]["depth_median"]
            for t in ["nose", "mouth", "eyes", "eyel", "eyer"]:
                if t in tag2pinfo and tag2pinfo[t]["depth_median"] > face_dm:
                    tag2pinfo[t]["depth_median"] = face_dm - 0.001
            for t in ["earr", "earl", "ears"]:
                if t in tag2pinfo:
                    tag2pinfo[t]["depth_median"] = face_dm + 0.001

        frame_size = fullpage.shape[:2]
        parts_data = {"tag2pinfo": tag2pinfo, "frame_size": frame_size}

        print(f"[SeeThrough] PostProcess complete: {len(tag2pinfo)} layers", flush=True)
        for tag, pinfo in sorted(tag2pinfo.items(), key=lambda x: x[1].get("depth_median", 1)):
            dm = pinfo.get("depth_median", "?")
            print(f"  - {tag}: depth_median={dm:.4f}" if isinstance(dm, float) else f"  - {tag}", flush=True)

        preview = _make_preview(tag2pinfo, resolution)
        return (parts_data, preview)


class SeeThrough_SavePSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
                "filename_prefix": ("STRING", {
                    "default": "seethrough",
                    "tooltip": "Output filename prefix. The frontend sets this automatically from the "
                               "LoadImage filename (TT.png -> TT), so the folder name, the standing "
                               "image name and this prefix all match.",
                }),
            },
            "optional": {
                "save_depth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write <layer>_depth.png next to every layer. Off by default: "
                               "depth is only needed for parallax/3D export, and it doubles the "
                               "file count in the output folder. Drawing order (depth_median) is "
                               "always recorded in layers.json either way.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_file",)
    FUNCTION = "save"
    CATEGORY = "SeeThrough"
    OUTPUT_NODE = True

    def save(self, parts, filename_prefix="seethrough", save_depth=False):
        from PIL import Image
        import json

        # 旧工作流里这个位置可能是前端按钮留下的字符串（"Download PSD"），
        # 直接 bool() 会把它当真。只认明确的真值写法。
        save_depth = _as_bool(save_depth, False)

        tag2pinfo = parts["tag2pinfo"]
        frame_size = parts["frame_size"]
        canvas_h, canvas_w = frame_size

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 文件名保持「前缀_时间戳_部位」，不再带 8 位随机后缀：一轮 29 层的名字短一半，
        # 下游按时间戳认批次也简单（20261024_151110 而不是 20261024_151110_7291cdfe）。
        # 同一秒里跑两次才可能撞（一轮要几分钟），撞了才补 4 位随机数。
        if os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{ts}_layers.json")):
            ts = f"{ts}_{str(uuid.uuid4())[:4]}"

        sorted_tags = sorted(tag2pinfo.keys(), key=lambda t: tag2pinfo[t].get("depth_median", 1), reverse=True)

        layer_info_list = []
        for tag in sorted_tags:
            pinfo = tag2pinfo[tag]
            img = pinfo.get("img")
            depth = pinfo.get("depth")
            if img is None:
                continue

            xyxy = pinfo.get("xyxy", [0, 0, img.shape[1], img.shape[0]])
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            layer_filename = f"{filename_prefix}_{ts}_{tag}.png"
            Image.fromarray(img).save(os.path.join(output_dir, layer_filename))

            entry = {"name": tag, "filename": layer_filename,
                     "left": x1, "top": y1, "right": x2, "bottom": y2,
                     "depth_median": float(pinfo.get("depth_median", 1))}

            if save_depth and depth is not None:
                depth_filename = f"{filename_prefix}_{ts}_{tag}_depth.png"
                if depth.ndim == 2:
                    Image.fromarray(depth, mode="L").save(os.path.join(output_dir, depth_filename))
                else:
                    Image.fromarray(depth).save(os.path.join(output_dir, depth_filename))
                entry["depth_filename"] = depth_filename

            layer_info_list.append(entry)

        info_filename = f"{filename_prefix}_{ts}_layers.json"
        info_path = os.path.join(output_dir, info_filename)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"prefix": filename_prefix, "timestamp": ts,
                       "layers": layer_info_list, "width": int(canvas_w), "height": int(canvas_h)}, f, indent=2)

        log_path = os.path.join(output_dir, "seethrough_psd_info.log")
        with open(log_path, "w") as f:
            f.write(info_filename)

        print(f"[SeeThrough] {len(layer_info_list)} layers saved. Use 'Download PSD' button to generate PSD.", flush=True)
        return (info_path,)


class SeeThrough_PartsToLayers:
    """Convert SEETHROUGH_PARTS into a core LAYERS document for the built-in
    compositor / layer editor (Create Layered Image node)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
            },
        }

    RETURN_TYPES = ("LAYERS",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "convert"
    CATEGORY = "SeeThrough"

    def convert(self, parts):
        tag2pinfo = parts["tag2pinfo"]
        canvas_h, canvas_w = parts["frame_size"]

        sorted_tags = sorted(tag2pinfo.keys(),
                             key=lambda t: tag2pinfo[t].get("depth_median", 1), reverse=True)

        items = []
        for z_index, tag in enumerate(sorted_tags):
            pinfo = tag2pinfo[tag]
            img = pinfo.get("img")
            if img is None:
                continue
            xyxy = pinfo.get("xyxy", [0, 0, img.shape[1], img.shape[0]])
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            items.append({
                "image": tensor,
                "type": "raster",
                "name": tag,
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "z_index": z_index,
            })

        document = {
            "version": 1,
            "layers": items,
            "canvas": (int(canvas_w), int(canvas_h)),
        }
        return (document,)


NODE_CLASS_MAPPINGS = {
    "SeeThrough_LoadLayerDiffModel": SeeThrough_LoadLayerDiffModel,
    "SeeThrough_LoadDepthModel": SeeThrough_LoadDepthModel,
    "SeeThrough_GenerateLayers": SeeThrough_GenerateLayers,
    "SeeThrough_GenerateDepth": SeeThrough_GenerateDepth,
    "SeeThrough_PostProcess": SeeThrough_PostProcess,
    "SeeThrough_SavePSD": SeeThrough_SavePSD,
    "SeeThrough_PartsToLayers": SeeThrough_PartsToLayers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeeThrough_LoadLayerDiffModel": "SeeThrough Load LayerDiff Model",
    "SeeThrough_LoadDepthModel": "SeeThrough Load Depth Model",
    "SeeThrough_GenerateLayers": "SeeThrough Generate Layers",
    "SeeThrough_GenerateDepth": "SeeThrough Generate Depth",
    "SeeThrough_PostProcess": "SeeThrough Post Process",
    "SeeThrough_SavePSD": "SeeThrough Save PSD",
    "SeeThrough_PartsToLayers": "SeeThrough Parts To Layers",
}
