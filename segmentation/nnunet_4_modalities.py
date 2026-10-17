# -*- coding: utf-8 -*-
"""nnUnet_4_modalities.ipynb
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Install Libraries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Install all required libraries.
#   DynUNet (nnU-Net) is built into MONAI — no extra package needed.
#   Identical to all other model notebooks.
# ─────────────────────────────────────────────────────────────────────────────

!pip install monai nibabel torchinfo -q

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='monai')

import monai

print("=" * 55)
print("CELL 1 — LIBRARIES INSTALLED")
print("=" * 55)
print("  monai     ✓  Medical imaging framework")
print("  nibabel   ✓  NIfTI file reader")
print("  torchinfo ✓  Model summary printer")
print(f"  MONAI version : {monai.__version__}")
print("  DynUNet (nnU-Net) is built into MONAI ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Mount Google Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from google.colab import drive
drive.mount('/content/drive')

print("=" * 55)
print("CELL 2 — GOOGLE DRIVE MOUNTED")
print("=" * 55)
print("  Mount point : /content/drive/MyDrive")
print("  Status      : Connected ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Keep Alive Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from IPython.display import display, Javascript

def keep_alive():
    display(Javascript('''
        function click_connect(){
            console.log("Keeping Colab alive...");
            document.querySelector("#top-toolbar > colab-connect-button")
                    .shadowRoot
                    .querySelector("#connect")
                    .click();
        }
        setInterval(click_connect, 60000);
    '''))

keep_alive()
print("=" * 55)
print("CELL 3 — KEEP ALIVE RUNNING ✓")
print("=" * 55)
print("  Clicks connect every 60 seconds ✓")
print("  Prevents idle disconnect ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — All Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Centralise every import.
#   Key difference from other models:
#     DynUNet — MONAI implementation of nnU-Net architecture.
#   All other imports identical for fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

import os, glob, time, random, gc
import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim              import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo                import summary

from monai.transforms import (
    Compose, LoadImaged,
    EnsureChannelFirstd, EnsureTyped,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d,
    MapTransform, AsDiscrete,
    Lambda,
)
from monai.data import (
    Dataset    as MonaiDataset,
    DataLoader as MonaiDataLoader,
)
from monai.networks.nets  import DynUNet          # nnU-Net implementation
from monai.losses         import DiceCELoss
from monai.metrics        import (
    DiceMetric, MeanIoU,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.inferers import sliding_window_inference
from monai.utils    import set_determinism

# Reproducibility
set_determinism(seed=42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 55)
print("CELL 4 — ENVIRONMENT")
print("=" * 55)
print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {device}")
if torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  GPU RAM  : "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  ⚠️  GPU not active — Runtime → GPU → Save")
print("  Seed     : 42 ✓")
print("  DynUNet  : imported from MONAI ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Paths & Global Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define all paths and constants.
#   Key differences from other models:
#     MODEL_NAME = 'nnUNet'    — identifies model in all CSVs
#     SAVE_DIR   = .../nnunet/ — separate folder from other models
#   VAL_INTERVAL=4, CKPT_INTERVAL=4 — same as all other models.
# ─────────────────────────────────────────────────────────────────────────────

BASE     = '/content/drive/MyDrive/Research/Implementation/BRATS_DATASET/'
DATA_DIR = BASE + 'MICCAI_BraTS2020_TrainingData/'
SAVE_DIR = BASE + 'pytorch_outputs/nnunet/'
CKPT_DIR = SAVE_DIR + 'checkpoints/'
LOG_DIR  = SAVE_DIR + 'logs/'

LOG_TRAIN = LOG_DIR + 'train_log.csv'
LOG_VAL   = LOG_DIR + 'val_log.csv'
LOG_FULL  = LOG_DIR + 'full_log.csv'
LOG_EFF   = LOG_DIR + 'efficiency_log.csv'

MODEL_NAME    = 'nnUNet'
DATASET_TYPE  = 'BraTS2020'
MAX_EPOCHS    = 100
VAL_INTERVAL  = 4
CKPT_INTERVAL = 4

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

cases = sorted(glob.glob(DATA_DIR + 'BraTS20_Training_*'))

print("=" * 55)
print("CELL 5 — PATHS & CONSTANTS")
print("=" * 55)
print(f"  DATA_DIR exists : {os.path.exists(DATA_DIR)}")
print(f"  Total cases     : {len(cases)}")
print(f"  MODEL_NAME      : {MODEL_NAME}")
print(f"  DATASET_TYPE    : {DATASET_TYPE}")
print(f"  MAX_EPOCHS      : {MAX_EPOCHS}")
print(f"  VAL_INTERVAL    : {VAL_INTERVAL}  (every 4 epochs)")
print(f"  CKPT_INTERVAL   : {CKPT_INTERVAL}  (every 4 epochs)")
print()
print(f"  SAVE_DIR  : {SAVE_DIR}")
print(f"  CKPT_DIR  : {CKPT_DIR}")
print(f"  LOG_DIR   : {LOG_DIR}")
print()
print("  Separate folder from all other models ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Data Dictionaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Identical to all other models.
#   Same four modalities — FLAIR, T1, T1ce, T2.
# ─────────────────────────────────────────────────────────────────────────────

def find_nii(path):
    if os.path.exists(path + '.gz'):
        return path + '.gz'
    if os.path.exists(path):
        return path
    return None


def create_data_dicts(data_dir):
    all_cases  = sorted(glob.glob(data_dir + 'BraTS20_Training_*'))
    data_dicts = []
    skipped    = []
    for case in all_cases:
        name  = os.path.basename(case)
        flair = find_nii(os.path.join(case, f'{name}_flair.nii'))
        t1    = find_nii(os.path.join(case, f'{name}_t1.nii'))
        t1ce  = find_nii(os.path.join(case, f'{name}_t1ce.nii'))
        t2    = find_nii(os.path.join(case, f'{name}_t2.nii'))
        seg   = find_nii(os.path.join(case, f'{name}_seg.nii'))
        if all(f is not None for f in [flair, t1, t1ce, t2, seg]):
            data_dicts.append({
                'image' : [flair, t1, t1ce, t2],
                'label' : seg,
                'name'  : name,
            })
        else:
            skipped.append(name)
            missing = [k for k, v in zip(
                ['flair','t1','t1ce','t2','seg'],
                [flair, t1, t1ce, t2, seg]) if v is None]
            print(f"  ⚠️  Skipped {name}  missing: {missing}")
    return data_dicts, skipped


all_data, skipped = create_data_dicts(DATA_DIR)

if len(all_data) == 0:
    raise ValueError("No valid cases found. Check DATA_DIR.")

d = all_data[0]
print("=" * 55)
print("CELL 6 — DATA DICTIONARIES CREATED")
print("=" * 55)
print(f"  Valid   : {len(all_data)} cases ✓")
print(f"  Skipped : {len(skipped)} cases")
print()
print(f"    image[0] : FLAIR")
print(f"    image[1] : T1")
print(f"    image[2] : T1ce")
print(f"    image[3] : T2")
print(f"    label    : Segmentation")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Train / Val Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   80/20 patient-level split with seed=42.
#   Identical to all other models for fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

random.seed(42)
shuffled   = all_data.copy()
random.shuffle(shuffled)
split      = int(len(shuffled) * 0.80)
train_data = shuffled[:split]
val_data   = shuffled[split:]

print("=" * 55)
print("CELL 7 — TRAIN / VAL SPLIT")
print("=" * 55)
print(f"  Total  : {len(all_data)}")
print(f"  Train  : {len(train_data)}  (80%)")
print(f"  Val    : {len(val_data)}  (20%)")
print(f"  Seed   : 42 — same as all other models ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Identical to all other models for fair comparison.
#   Same crop size 128³, same augmentation, same label remap 4→3.
# ─────────────────────────────────────────────────────────────────────────────

class ConvertBraTSLabels(MapTransform):
    """Remap BraTS 2020 label 4→3 for contiguous class indices."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            result = (torch.zeros_like(label)
                      if isinstance(label, torch.Tensor)
                      else np.zeros_like(label))
            result[label == 1] = 1
            result[label == 2] = 2
            result[label == 4] = 3
            d[key] = result
        return d


def unwrap_crop_list(data):
    if isinstance(data, list):
        return data[0]
    return data


train_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    ConvertBraTSLabels(keys=['label']),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys         = ['image', 'label'],
        label_key    = 'label',
        spatial_size = (128, 128, 128),
        pos          = 1,
        neg          = 1,
        num_samples  = 1,
    ),
    Lambda(unwrap_crop_list),
    RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.5),
    RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),
    RandFlipd(keys=['image', 'label'], spatial_axis=2, prob=0.5),
    RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
    EnsureTyped(keys=['image', 'label']),
])

val_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    ConvertBraTSLabels(keys=['label']),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    EnsureTyped(keys=['image', 'label']),
])

print("=" * 55)
print("CELL 8 — TRANSFORMS DEFINED")
print("=" * 55)
print("  Label remap  : 4 → 3  ✓")
print("  Crop size    : 128³   ✓")
print("  Augmentation : flip×3 + rotate90  ✓")
print("  Identical to all other models ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Datasets + DataLoaders + Sample Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_dataset = MonaiDataset(data=train_data, transform=train_transforms)
val_dataset   = MonaiDataset(data=val_data,   transform=val_transforms)

train_loader = MonaiDataLoader(
    train_dataset, batch_size=1,
    shuffle=True, num_workers=2, pin_memory=True)
val_loader = MonaiDataLoader(
    val_dataset, batch_size=1,
    shuffle=False, num_workers=2, pin_memory=True)

print("=" * 55)
print("CELL 9 — DATASETS + DATALOADERS")
print("=" * 55)
print(f"  Train dataset : {len(train_dataset)}")
print(f"  Val   dataset : {len(val_dataset)}")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val   batches : {len(val_loader)}")

# Sample verification
sample = train_dataset[0]
if isinstance(sample, list):
    sample = sample[0]
img_shape  = sample['image'].shape
lbl_shape  = sample['label'].shape
unique_lbs = torch.unique(sample['label']).tolist()
assert img_shape[0] == 4,  f"Expected 4 channels, got {img_shape[0]}"
assert lbl_shape[0] == 1,  f"Expected 1 label ch, got {lbl_shape[0]}"
assert set(unique_lbs).issubset({0.0,1.0,2.0,3.0}), \
    f"Unexpected labels: {unique_lbs}"
print(f"  Image shape   : {list(img_shape)}")
print(f"  Label values  : {unique_lbs}")
print(f"  ✅ 4 channels confirmed")
print(f"  ✅ Label remap 4→3 applied")
print(f"  ✅ DataLoaders ready")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 — Build nnU-Net (DynUNet) Model, Optimiser, Scheduler, Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Instantiate DynUNet — MONAI's implementation of nnU-Net.
#
#   nnU-Net Architecture (Nature Methods 2021):
#     Self-configuring framework — automatically determines optimal
#     architecture, preprocessing, and training from dataset properties.
#     DynUNet implements the core nnU-Net encoder-decoder with:
#       Dynamic kernel sizes per stage
#       Residual blocks for stable gradient flow
#       Deep supervision during training (disabled here for fairness)
#       Anisotropic convolutions for non-isotropic data
#
#   Configuration for BraTS 2020 (128³ isotropic):
#     5 resolution stages
#     Kernels: 3×3×3 at all stages (isotropic)
#     Strides: 1 at stage 0, 2×2×2 for stages 1-4 (downsampling)
#     in_channels  = 4   FLAIR, T1, T1ce, T2
#     out_channels = 4   BG, NCR, Edema, ET
#     norm_name    = batch
#     res_block    = True  residual connections
#
#   Same optimiser, scheduler, loss as all other models
#   for completely fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

# Clear GPU before loading model
gc.collect()
torch.cuda.empty_cache()
alloc = torch.cuda.memory_allocated(device)/1024/1024
print(f"  GPU before model load : {alloc:.1f} MB ✓")

# nnU-Net kernel and stride configuration for 128³ input
kernels = [
    [3, 3, 3],   # stage 0 — full resolution
    [3, 3, 3],   # stage 1
    [3, 3, 3],   # stage 2
    [3, 3, 3],   # stage 3
    [3, 3, 3],   # stage 4 — bottleneck
]
strides = [
    [1, 1, 1],   # stage 0 — no downsampling
    [2, 2, 2],   # stage 1 — 128→64
    [2, 2, 2],   # stage 2 — 64→32
    [2, 2, 2],   # stage 3 — 32→16
    [2, 2, 2],   # stage 4 — 16→8
]

model = DynUNet(
    spatial_dims         = 3,
    in_channels          = 4,          # FLAIR, T1, T1ce, T2
    out_channels         = 4,          # BG, NCR, Edema, ET
    kernel_size          = kernels,
    strides              = strides,
    upsample_kernel_size = strides[1:],  # decoder strides
    norm_name            = 'batch',
    deep_supervision     = False,        # off for fair comparison
    res_block            = True,         # residual connections
).to(device)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

loss_fn = DiceCELoss(
    include_background = False,
    to_onehot_y        = True,
    softmax            = True,
    lambda_dice        = 0.5,
    lambda_ce          = 0.5,
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
non_trainable    = total_params - trainable_params

gc.collect()
torch.cuda.empty_cache()
alloc = torch.cuda.memory_allocated(device)/1024/1024

print("=" * 55)
print("CELL 10 — MODEL READY — nnU-Net (DynUNet)")
print("=" * 55)
print(f"  Architecture     : nnU-Net / DynUNet (Nature Methods 2021)")
print(f"  Dataset          : {DATASET_TYPE}")
print()
print("  DynUNet Configuration:")
print(f"    spatial_dims   : 3   (volumetric)")
print(f"    in_channels    : 4   (FLAIR, T1, T1ce, T2)")
print(f"    out_channels   : 4   (BG, NCR, Edema, ET)")
print(f"    kernel_size    : 3×3×3 at all 5 stages")
print(f"    strides        : 1,2,2,2,2  (5 stages)")
print(f"    norm_name      : batch")
print(f"    res_block      : True")
print(f"    deep_supervision: False  (fair comparison)")
print()
print("  Encoder shape flow:")
print(f"    Stage 0 :   4 ch  128³ → 128³  (no downsample)")
print(f"    Stage 1 :  32 ch  128³ →  64³")
print(f"    Stage 2 :  64 ch   64³ →  32³")
print(f"    Stage 3 : 128 ch   32³ →  16³")
print(f"    Stage 4 : 256 ch   16³ →   8³  (bottleneck)")
print()
print("  Training:")
print(f"    Optimizer      : Adam  lr=1e-4  wd=1e-5")
print(f"    Scheduler      : CosineAnnealingLR T_max={MAX_EPOCHS}")
print(f"    Loss           : DiceCELoss  0.5+0.5")
print()
print("  Parameters:")
print(f"    Total          : {total_params:,}")
print(f"    Trainable      : {trainable_params:,}")
print(f"    Non-trainable  : {non_trainable:,}")
print(f"    Params (M)     : {total_params/1e6:.4f} M")
print(f"    Model size     : {total_params*4/1024/1024:.2f} MB")
print(f"    GPU after load : {alloc:.1f} MB")
print(f"    Device         : {device}")
print()
print("  Model ready ✓")
print("  Ready for Cell 11 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — Measure Efficiency Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def measure_efficiency(model, device,
                       input_shape=(1, 4, 128, 128, 128),
                       n_warmup=3, n_runs=10):
    model.eval()
    dummy        = torch.zeros(*input_shape).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                starter.record()
                model(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        inference_ms  = np.mean(timings)
        gpu_memory_MB = (torch.cuda.max_memory_allocated(device)
                         / 1024 / 1024)
    else:
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.time()
                model(dummy)
                timings.append((time.time()-t0)*1000)
        inference_ms  = np.mean(timings)
        gpu_memory_MB = 0.0

    del dummy
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return {
        'total_params'  : total_params,
        'params_M'      : round(total_params/1e6,            4),
        'model_size_MB' : round(total_params*4/1024/1024,    2),
        'inference_ms'  : round(inference_ms,                2),
        'gpu_memory_MB' : round(gpu_memory_MB,               2),
    }


print("=" * 55)
print("CELL 11 — EFFICIENCY METRICS")
print("=" * 55)
print(f"  Model      : {MODEL_NAME}")
print(f"  Warmup     : 3 runs")
print(f"  Timed      : 10 runs")
print()
print("  Measuring...")
print()

eff = measure_efficiency(model, device)

print(f"  Parameters     : {eff['total_params']:,}")
print(f"  Parameters (M) : {eff['params_M']:.4f} M")
print(f"  Model size     : {eff['model_size_MB']:.2f} MB")
print(f"  Inference time : {eff['inference_ms']:.2f} ms / sample")
print(f"  GPU memory     : {eff['gpu_memory_MB']:.2f} MB")
print()

# Verify GPU is clear after measurement
alloc = torch.cuda.memory_allocated(device)/1024/1024
print(f"  GPU after measurement : {alloc:.1f} MB")
if alloc < 1000:
    print("  ✅ GPU RAM clear — safe for training ✓")
else:
    print("  ⚠️  GPU RAM high — run gc.collect()+empty_cache()")
print()
print("  Efficiency values ready for paper table ✓")
print("  Ready for Cell 12 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — Model Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 55)
print("CELL 12 — MODEL SUMMARY")
print("=" * 55)
print(f"  Model   : {MODEL_NAME}")
print(f"  Params  : {eff['params_M']:.4f} M")
print(f"  Size    : {eff['model_size_MB']:.2f} MB")
print(f"  Infer   : {eff['inference_ms']:.2f} ms / sample")
print(f"  GPU mem : {eff['gpu_memory_MB']:.2f} MB")
print()
print("  Generating layer-by-layer summary...")
print()

dummy_input = torch.zeros(1, 4, 128, 128, 128).to(device)
model_summary = summary(
    model,
    input_data   = dummy_input,
    col_names    = ['input_size', 'output_size',
                    'num_params', 'trainable'],
    col_width    = 20,
    row_settings = ['var_names'],
    verbose      = 0,
)
print(model_summary)

del dummy_input
gc.collect()
torch.cuda.empty_cache()
print("  Summary printed ✓")
print("  GPU cache cleared ✓")
print("  Ready for Cell 13 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — Define All Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

dice_metric = DiceMetric(
    include_background=False, reduction='mean_batch', get_not_nans=True)
iou_metric = MeanIoU(
    include_background=False, reduction='mean_batch', get_not_nans=True)
confusion_metric = ConfusionMatrixMetric(
    include_background=False,
    metric_name=['sensitivity', 'specificity', 'precision'],
    reduction='mean_batch', get_not_nans=True)
dice_wt = DiceMetric(
    include_background=True, reduction='mean', get_not_nans=True)
dice_tc = DiceMetric(
    include_background=True, reduction='mean', get_not_nans=True)
dice_et = DiceMetric(
    include_background=True, reduction='mean', get_not_nans=True)

post_pred  = AsDiscrete(argmax=True, to_onehot=4)
post_label = AsDiscrete(to_onehot=4)

print("=" * 55)
print("CELL 13 — METRICS DEFINED")
print("=" * 55)
print(f"  Model   : {MODEL_NAME}")
print(f"  Dataset : {DATASET_TYPE}")
print()
print("  Primary  (include_background=False):")
print("    DiceMetric      ✓")
print("    MeanIoU         ✓")
print("    ConfusionMatrix ✓  sens, spec, prec")
print()
print("  BraTS sub-region  (include_background=True):")
print("    Dice WT  ✓  NCR+Edema+ET")
print("    Dice TC  ✓  NCR+ET")
print("    Dice ET  ✓  ET only")
print()
print("  HD95 : final eval cell only")
print("  Identical to all other models ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14 — Training & Validation Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_subregions(pred_onehot, label_onehot):
    assert pred_onehot.ndim == 4
    assert pred_onehot.shape[0] == 4
    pred_wt  = (pred_onehot[1]+pred_onehot[2]+pred_onehot[3]).clamp(0,1)
    label_wt = (label_onehot[1]+label_onehot[2]+label_onehot[3]).clamp(0,1)
    pred_tc  = (pred_onehot[1]+pred_onehot[3]).clamp(0,1)
    label_tc = (label_onehot[1]+label_onehot[3]).clamp(0,1)
    pred_et  = pred_onehot[3]
    label_et = label_onehot[3]
    def wrap(t):
        return t.unsqueeze(0).unsqueeze(0)
    return (wrap(pred_wt), wrap(label_wt),
            wrap(pred_tc), wrap(label_tc),
            wrap(pred_et), wrap(label_et))


def extract_conf(result, idx):
    val = result[idx]
    if isinstance(val, tuple):
        tensor = val[0]
        return (tensor.mean().item()
                if isinstance(tensor, torch.Tensor)
                else float(tensor))
    if isinstance(val, torch.Tensor):
        return val.mean().item()
    return float(val)


def train_one_epoch(model, loader, optimizer,
                    loss_fn, device, epoch):
    model.train()
    epoch_loss  = 0.0
    steps       = 0
    epoch_start = time.time()
    for batch_idx, batch in enumerate(loader):
        steps  += 1
        images  = batch['image'].to(device)
        labels  = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"    [{batch_idx+1:>4}/{len(loader)}]"
                  f"  Loss: {loss.item():.4f}")
    return epoch_loss/steps, (time.time()-epoch_start)/60.0


def validate(model, loader, loss_fn,
             dice_metric, iou_metric,
             confusion_metric,
             dice_wt, dice_tc, dice_et,
             post_pred, post_label, device):
    model.eval()
    val_loss = 0.0
    steps    = 0
    with torch.no_grad():
        for batch in loader:
            steps  += 1
            images  = batch['image'].to(device)
            labels  = batch['label'].to(device)
            outputs = sliding_window_inference(
                images,
                roi_size      = (128, 128, 128),
                sw_batch_size = 4,
                predictor     = model,
                overlap       = 0.5,
                mode          = 'gaussian',
            )
            val_loss += loss_fn(outputs, labels).item()
            outputs_post = [post_pred(o)  for o in outputs]
            labels_post  = [post_label(l) for l in labels]
            dice_metric(y_pred=outputs_post, y=labels_post)
            iou_metric( y_pred=outputs_post, y=labels_post)
            confusion_metric(y_pred=outputs_post, y=labels_post)
            for pred_oh, label_oh in zip(outputs_post, labels_post):
                p_wt,l_wt,p_tc,l_tc,p_et,l_et = \
                    extract_subregions(pred_oh, label_oh)
                dice_wt(y_pred=p_wt, y=l_wt)
                dice_tc(y_pred=p_tc, y=l_tc)
                dice_et(y_pred=p_et, y=l_et)

    def get_val(result):
        if isinstance(result, tuple):
            return result[0].mean().item()
        return result.mean().item()

    val_dice = get_val(dice_metric.aggregate())
    val_iou  = get_val(iou_metric.aggregate())
    conf_result = confusion_metric.aggregate()
    if isinstance(conf_result, (list, tuple)):
        val_sens = extract_conf(conf_result, 0)
        val_spec = extract_conf(conf_result, 1)
        val_prec = extract_conf(conf_result, 2)
    else:
        val_sens = conf_result[...,0].mean().item()
        val_spec = conf_result[...,1].mean().item()
        val_prec = conf_result[...,2].mean().item()
    val_wt = get_val(dice_wt.aggregate())
    val_tc = get_val(dice_tc.aggregate())
    val_et = get_val(dice_et.aggregate())
    for m in [dice_metric, iou_metric, confusion_metric,
              dice_wt, dice_tc, dice_et]:
        m.reset()
    return {
        'val_loss': val_loss/steps,  'val_dice': val_dice,
        'val_iou' : val_iou,         'val_hd95': float('nan'),
        'val_sens': val_sens,        'val_spec': val_spec,
        'val_prec': val_prec,        'val_wt'  : val_wt,
        'val_tc'  : val_tc,          'val_et'  : val_et,
    }


print("=" * 55)
print("CELL 14 — FUNCTIONS DEFINED ✓")
print("=" * 55)
print("  train_one_epoch    ✓")
print("  validate           ✓")
print("  extract_subregions ✓")
print("  extract_conf       ✓")
print("  sw_batch_size=4    ✓  safe for nnU-Net")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15 — Define save_logs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_logs(history, log_train, log_val,
              log_full, log_eff, eff_static):

    pd.DataFrame({
        'epoch'          : history['epoch'],
        'train_loss'     : history['train_loss'],
        'lr'             : history['lr'],
        'train_time_min' : history['train_time_min'],
    }).to_csv(log_train, index=False)

    if len(history['val_loss']) > 0:
        pd.DataFrame({
            'epoch'    : history['val_epoch'],
            'val_loss' : history['val_loss'],
            'val_dice' : history['val_dice'],
            'val_iou'  : history['val_iou'],
            'val_hd95' : history['val_hd95'],
            'val_sens' : history['val_sens'],
            'val_spec' : history['val_spec'],
            'val_prec' : history['val_prec'],
            'val_wt'   : history['val_wt'],
            'val_tc'   : history['val_tc'],
            'val_et'   : history['val_et'],
        }).to_csv(log_val, index=False)

    val_lookup = {ve: i for i, ve in enumerate(history['val_epoch'])}
    full_rows  = []
    for i, ep in enumerate(history['epoch']):
        row = {
            'epoch'          : ep,
            'train_loss'     : history['train_loss'][i],
            'lr'             : history['lr'][i],
            'train_time_min' : history['train_time_min'][i],
            'val_loss': None, 'val_dice': None,
            'val_iou' : None, 'val_hd95': None,
            'val_sens': None, 'val_spec': None,
            'val_prec': None, 'val_wt'  : None,
            'val_tc'  : None, 'val_et'  : None,
        }
        if ep in val_lookup:
            vi = val_lookup[ep]
            row.update({
                'val_loss': history['val_loss'][vi],
                'val_dice': history['val_dice'][vi],
                'val_iou' : history['val_iou'][vi],
                'val_hd95': history['val_hd95'][vi],
                'val_sens': history['val_sens'][vi],
                'val_spec': history['val_spec'][vi],
                'val_prec': history['val_prec'][vi],
                'val_wt'  : history['val_wt'][vi],
                'val_tc'  : history['val_tc'][vi],
                'val_et'  : history['val_et'][vi],
            })
        full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(log_full, index=False)

    mean_time = round(np.mean(history['train_time_min']), 4)
    eff_rows  = [{
        'model'          : MODEL_NAME,
        'dataset'        : DATASET_TYPE,
        'params_M'       : eff_static['params_M'],
        'model_size_MB'  : eff_static['model_size_MB'],
        'inference_ms'   : eff_static['inference_ms'],
        'gpu_memory_MB'  : eff_static['gpu_memory_MB'],
        'epoch'          : ep,
        'train_time_min' : history['train_time_min'][i],
    } for i, ep in enumerate(history['epoch'])]

    summary_row = pd.DataFrame([{
        'model'          : MODEL_NAME,
        'dataset'        : DATASET_TYPE,
        'params_M'       : eff_static['params_M'],
        'model_size_MB'  : eff_static['model_size_MB'],
        'inference_ms'   : eff_static['inference_ms'],
        'gpu_memory_MB'  : eff_static['gpu_memory_MB'],
        'epoch'          : 'MEAN',
        'train_time_min' : mean_time,
    }])

    pd.concat([pd.DataFrame(eff_rows), summary_row],
              ignore_index=True).to_csv(log_eff, index=False)

print("=" * 55)
print("CELL 15 — SAVE_LOGS DEFINED ✓")
print("=" * 55)
print(f"  model column   : {MODEL_NAME}")
print(f"  dataset column : {DATASET_TYPE}")
print("  Stackable with all other model CSVs ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 16 — Training Loop (100 Epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gc

try:
    START_EPOCH
    print(f"  Resuming from epoch {START_EPOCH}")
except NameError:
    START_EPOCH = 1
    best_dice   = 0.0
    history = {
        'epoch':[], 'train_loss':[], 'lr':[],
        'train_time_min':[], 'val_epoch':[],
        'val_loss':[], 'val_dice':[], 'val_iou':[],
        'val_hd95':[], 'val_sens':[], 'val_spec':[],
        'val_prec':[], 'val_wt':[], 'val_tc':[], 'val_et':[],
    }
    print("  Fresh training from epoch 1")

best_epoch = 0

gc.collect()
torch.cuda.empty_cache()
alloc = torch.cuda.memory_allocated(device)/1024/1024
print(f"  GPU before training : {alloc:.1f} MB")
print()

print("=" * 55)
print("CELL 16 — STARTING TRAINING — nnU-Net")
print("=" * 55)
print(f"  Model         : {MODEL_NAME}")
print(f"  Dataset       : {DATASET_TYPE}")
print(f"  Device        : {device}")
print(f"  Start epoch   : {START_EPOCH}")
print(f"  Max epochs    : {MAX_EPOCHS}")
print(f"  Val interval  : every {VAL_INTERVAL} epochs")
print(f"  Ckpt interval : every {CKPT_INTERVAL} epochs")
print(f"  Params        : {eff['params_M']:.4f} M")
print(f"  Model size    : {eff['model_size_MB']:.2f} MB")
print(f"  Checkpoints → {CKPT_DIR}")
print(f"  Logs        → {LOG_DIR}")
print("=" * 55)

for epoch in range(START_EPOCH, MAX_EPOCHS + 1):

    print(f"\n{'=' * 55}")
    print(f"  Epoch {epoch}/{MAX_EPOCHS}"
          f"  [{MODEL_NAME} — {DATASET_TYPE}]")
    print(f"  {'-' * 51}")

    train_loss, epoch_time = train_one_epoch(
        model, train_loader,
        optimizer, loss_fn,
        device, epoch)

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['lr'].append(lr)
    history['train_time_min'].append(round(epoch_time, 4))

    print(f"\n  Train Loss  : {train_loss:.4f}")
    print(f"  LR          : {lr:.7f}")
    print(f"  Epoch time  : {epoch_time:.2f} min")

    gc.collect()
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(device)/1024/1024
    print(f"  GPU after train : {alloc:.1f} MB")

    if epoch % VAL_INTERVAL == 0:

        print(f"\n  Running validation...")

        metrics = validate(
            model, val_loader, loss_fn,
            dice_metric, iou_metric,
            confusion_metric,
            dice_wt, dice_tc, dice_et,
            post_pred, post_label, device)

        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(device)/1024/1024
        print(f"  GPU after val   : {alloc:.1f} MB")

        history['val_epoch'].append(epoch)
        for k in ['val_loss','val_dice','val_iou','val_hd95',
                  'val_sens','val_spec','val_prec',
                  'val_wt','val_tc','val_et']:
            history[k].append(metrics[k])

        print(f"\n  ── Segmentation Metrics ─────────────────")
        print(f"  Val Loss    : {metrics['val_loss']:.4f}")
        print(f"  Dice        : {metrics['val_dice']:.4f}"
              f"  (Best: {max(history['val_dice']):.4f})")
        print(f"  IoU         : {metrics['val_iou']:.4f}")
        print(f"  HD95        : eval cell only (NaN in logs)")
        print(f"  Sensitivity : {metrics['val_sens']:.4f}")
        print(f"  Specificity : {metrics['val_spec']:.4f}")
        print(f"  Precision   : {metrics['val_prec']:.4f}")
        print(f"\n  ── BraTS Sub-regions ────────────────────")
        print(f"  Dice WT     : {metrics['val_wt']:.4f}")
        print(f"  Dice TC     : {metrics['val_tc']:.4f}")
        print(f"  Dice ET     : {metrics['val_et']:.4f}")
        print(f"\n  ── Efficiency ───────────────────────────")
        print(f"  Avg time    : "
              f"{np.mean(history['train_time_min']):.2f} min/epoch")
        print(f"  Total       : "
              f"{sum(history['train_time_min']):.2f} min")

        save_logs(history,
                  LOG_TRAIN, LOG_VAL,
                  LOG_FULL,  LOG_EFF, eff)
        print(f"\n  CSVs saved ✓")

        if metrics['val_dice'] > best_dice:
            best_dice  = metrics['val_dice']
            best_epoch = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_dice'  : best_dice,
                'metrics'    : metrics,
                'efficiency' : eff,
                'history'    : history,
                'model_name' : MODEL_NAME,
                'dataset'    : DATASET_TYPE,
            }, CKPT_DIR + 'best_model.pth')
            print(f"  ✓ Best model saved!"
                  f"  Dice={best_dice:.4f}  Epoch={best_epoch}")

    if epoch % CKPT_INTERVAL == 0:
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_dice'  : best_dice,
            'efficiency' : eff,
            'history'    : history,
            'model_name' : MODEL_NAME,
            'dataset'    : DATASET_TYPE,
        }, CKPT_DIR + f'epoch_{epoch:03d}.pth')
        print(f"  Checkpoint → epoch_{epoch:03d}.pth ✓")

save_logs(history, LOG_TRAIN, LOG_VAL, LOG_FULL, LOG_EFF, eff)

print(f"\n{'=' * 55}")
print(f"TRAINING COMPLETE — {MODEL_NAME}")
print("=" * 55)
print(f"  Best Dice  : {best_dice:.4f}  (Epoch {best_epoch})")
print(f"  Last WT    : {history['val_wt'][-1]:.4f}")
print(f"  Last TC    : {history['val_tc'][-1]:.4f}")
print(f"  Last ET    : {history['val_et'][-1]:.4f}")
print(f"  Params     : {eff['params_M']:.4f} M")
print(f"  Avg/epoch  : {np.mean(history['train_time_min']):.2f} min")
print(f"  Total time : {sum(history['train_time_min']):.2f} min")
print()
print(f"  Checkpoints : {CKPT_DIR}")
print(f"  Logs        : {LOG_DIR}")
print("\n  Next → Run Final Evaluation Cell ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINT STATUS CELL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import glob, os, torch, pandas as pd

print("=" * 55)
print("CHECKPOINT STATUS")
print("=" * 55)

checkpoints = sorted(glob.glob(CKPT_DIR + 'epoch_*.pth'))
print(f"  Epoch checkpoints : {len(checkpoints)}")
for ck in checkpoints:
    mb = os.path.getsize(ck) / 1024 / 1024
    print(f"    {os.path.basename(ck)}  {mb:.1f} MB")

print()

best_path = CKPT_DIR + 'best_model.pth'
if os.path.exists(best_path):
    ck = torch.load(best_path, map_location='cpu',
                    weights_only=False)
    print(f"  best_model.pth  ✓")
    print(f"    Epoch     : {ck['epoch']}")
    print(f"    Best Dice : {ck['best_dice']:.4f}")
    print(f"    Model     : {ck.get('model_name', 'N/A')}")
    print(f"    Dataset   : {ck.get('dataset', 'N/A')}")
else:
    print("  best_model.pth  ✗ not found")

print()

for log in [LOG_TRAIN, LOG_VAL, LOG_FULL, LOG_EFF]:
    if os.path.exists(log):
        df = pd.read_csv(log)
        print(f"  {os.path.basename(log)}"
              f"  —  {len(df)} rows ✓")
    else:
        print(f"  {os.path.basename(log)}  —  not found")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESUME CELL — nnU-Net BraTS2020
# Run after all setup cells on any disconnect
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gc

# Load eff from best_model.pth
best_path = CKPT_DIR + 'best_model.pth'
if os.path.exists(best_path):
    best_ck = torch.load(
        best_path,
        map_location = 'cpu',
        weights_only = False
    )
    eff = best_ck.get('efficiency', eff)
    print("  eff loaded from best_model.pth ✓")
else:
    print("  using eff from Cell 11 ✓")

# Find latest checkpoint automatically
all_ckpts = sorted(glob.glob(CKPT_DIR + 'epoch_*.pth'))

if len(all_ckpts) > 0:
    latest_ckpt = all_ckpts[-1]

    checkpoint = torch.load(
        latest_ckpt,
        map_location = device,
        weights_only = False
    )

    # Restore all components
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    history     = checkpoint['history']
    best_dice   = checkpoint['best_dice']
    START_EPOCH = checkpoint['epoch'] + 1

    # Restore scheduler to correct position
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max      = MAX_EPOCHS,
        eta_min    = 1e-6,
        last_epoch = checkpoint['epoch']
    )

    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 55)
    print("RESUMED — nnU-Net BraTS2020")
    print("=" * 55)
    print(f"  Checkpoint   : {os.path.basename(latest_ckpt)}")
    print(f"  Resumed from : epoch {checkpoint['epoch']}")
    print(f"  Starting at  : epoch {START_EPOCH}")
    print(f"  Remaining    : {MAX_EPOCHS - checkpoint['epoch']} epochs")
    print(f"  Best Dice    : {best_dice:.4f}")
    print(f"  History rows : {len(history['epoch'])} epochs")
    print(f"  LR           : {optimizer.param_groups[0]['lr']:.7f}")
    print()
    print("  model weights     ✓")
    print("  optimiser state   ✓")
    print("  history dict      ✓")
    print("  scheduler         ✓")
    print("  GPU cache cleared ✓")
    print()
    print("  Now run Cell 16 ✓")

else:
    START_EPOCH = 1
    best_dice   = 0.0
    history = {
        'epoch':[], 'train_loss':[], 'lr':[],
        'train_time_min':[], 'val_epoch':[],
        'val_loss':[], 'val_dice':[], 'val_iou':[],
        'val_hd95':[], 'val_sens':[], 'val_spec':[],
        'val_prec':[], 'val_wt':[], 'val_tc':[], 'val_et':[],
    }
    print("  No checkpoint found — starting from epoch 1")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 16 — Training Loop (100 Epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gc

try:
    START_EPOCH
    print(f"  Resuming from epoch {START_EPOCH}")
except NameError:
    START_EPOCH = 1
    best_dice   = 0.0
    history = {
        'epoch':[], 'train_loss':[], 'lr':[],
        'train_time_min':[], 'val_epoch':[],
        'val_loss':[], 'val_dice':[], 'val_iou':[],
        'val_hd95':[], 'val_sens':[], 'val_spec':[],
        'val_prec':[], 'val_wt':[], 'val_tc':[], 'val_et':[],
    }
    print("  Fresh training from epoch 1")

best_epoch = 0

gc.collect()
torch.cuda.empty_cache()
alloc = torch.cuda.memory_allocated(device)/1024/1024
print(f"  GPU before training : {alloc:.1f} MB")
print()

print("=" * 55)
print(f"CELL 16 — STARTING TRAINING")
print("=" * 55)
print(f"  Model         : {MODEL_NAME}")
print(f"  Dataset       : {DATASET_TYPE}")
print(f"  Device        : {device}")
print(f"  Start epoch   : {START_EPOCH}")
print(f"  Max epochs    : {MAX_EPOCHS}")
print(f"  Best Dice     : {best_dice:.4f}")
print(f"  Val interval  : every {VAL_INTERVAL} epochs")
print(f"  Ckpt interval : every {CKPT_INTERVAL} epochs")
print(f"  Params        : {eff['params_M']:.4f} M")
print(f"  Model size    : {eff['model_size_MB']:.2f} MB")
print(f"  Checkpoints → {CKPT_DIR}")
print(f"  Logs        → {LOG_DIR}")
print("=" * 55)

for epoch in range(START_EPOCH, MAX_EPOCHS + 1):

    print(f"\n{'=' * 55}")
    print(f"  Epoch {epoch}/{MAX_EPOCHS}"
          f"  [{MODEL_NAME} — {DATASET_TYPE}]")
    print(f"  {'-' * 51}")

    # 1. Train
    train_loss, epoch_time = train_one_epoch(
        model, train_loader,
        optimizer, loss_fn,
        device, epoch)

    # 2. LR step
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # 3. Record
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['lr'].append(lr)
    history['train_time_min'].append(round(epoch_time, 4))

    print(f"\n  Train Loss  : {train_loss:.4f}")
    print(f"  LR          : {lr:.7f}")
    print(f"  Epoch time  : {epoch_time:.2f} min")

    gc.collect()
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(device)/1024/1024
    print(f"  GPU after train : {alloc:.1f} MB")

    # 4. Validation
    if epoch % VAL_INTERVAL == 0:

        print(f"\n  Running validation...")

        metrics = validate(
            model, val_loader, loss_fn,
            dice_metric, iou_metric,
            confusion_metric,
            dice_wt, dice_tc, dice_et,
            post_pred, post_label, device)

        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(device)/1024/1024
        print(f"  GPU after val   : {alloc:.1f} MB")

        history['val_epoch'].append(epoch)
        for k in ['val_loss','val_dice','val_iou','val_hd95',
                  'val_sens','val_spec','val_prec',
                  'val_wt','val_tc','val_et']:
            history[k].append(metrics[k])

        print(f"\n  ── Segmentation Metrics ─────────────────")
        print(f"  Val Loss    : {metrics['val_loss']:.4f}")
        print(f"  Dice        : {metrics['val_dice']:.4f}"
              f"  (Best: {max(history['val_dice']):.4f})")
        print(f"  IoU         : {metrics['val_iou']:.4f}")
        print(f"  HD95        : eval cell only (NaN in logs)")
        print(f"  Sensitivity : {metrics['val_sens']:.4f}")
        print(f"  Specificity : {metrics['val_spec']:.4f}")
        print(f"  Precision   : {metrics['val_prec']:.4f}")
        print(f"\n  ── BraTS Sub-regions ────────────────────")
        print(f"  Dice WT     : {metrics['val_wt']:.4f}")
        print(f"  Dice TC     : {metrics['val_tc']:.4f}")
        print(f"  Dice ET     : {metrics['val_et']:.4f}")
        print(f"\n  ── Efficiency ───────────────────────────")
        print(f"  Avg time    : "
              f"{np.mean(history['train_time_min']):.2f} min/epoch")
        print(f"  Total       : "
              f"{sum(history['train_time_min']):.2f} min")

        save_logs(history,
                  LOG_TRAIN, LOG_VAL,
                  LOG_FULL,  LOG_EFF, eff)
        print(f"\n  CSVs saved ✓")

        if metrics['val_dice'] > best_dice:
            best_dice  = metrics['val_dice']
            best_epoch = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_dice'  : best_dice,
                'metrics'    : metrics,
                'efficiency' : eff,
                'history'    : history,
                'model_name' : MODEL_NAME,
                'dataset'    : DATASET_TYPE,
            }, CKPT_DIR + 'best_model.pth')
            print(f"  ✓ Best model saved!"
                  f"  Dice={best_dice:.4f}  Epoch={best_epoch}")

    # 5. Checkpoint
    if epoch % CKPT_INTERVAL == 0:
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_dice'  : best_dice,
            'efficiency' : eff,
            'history'    : history,
            'model_name' : MODEL_NAME,
            'dataset'    : DATASET_TYPE,
        }, CKPT_DIR + f'epoch_{epoch:03d}.pth')
        print(f"  Checkpoint → epoch_{epoch:03d}.pth ✓")

# Final save
save_logs(history, LOG_TRAIN, LOG_VAL, LOG_FULL, LOG_EFF, eff)

print(f"\n{'=' * 55}")
print(f"TRAINING COMPLETE — {MODEL_NAME}")
print("=" * 55)
print(f"  Best Dice  : {best_dice:.4f}  (Epoch {best_epoch})")
print(f"  Last WT    : {history['val_wt'][-1]:.4f}")
print(f"  Last TC    : {history['val_tc'][-1]:.4f}")
print(f"  Last ET    : {history['val_et'][-1]:.4f}")
print(f"  Params     : {eff['params_M']:.4f} M")
print(f"  Avg/epoch  : {np.mean(history['train_time_min']):.2f} min")
print(f"  Total time : {sum(history['train_time_min']):.2f} min")
print()
print(f"  Checkpoints : {CKPT_DIR}")
print(f"  Logs        : {LOG_DIR}")
print("\n  Next → Run Final Evaluation Cell ✓")