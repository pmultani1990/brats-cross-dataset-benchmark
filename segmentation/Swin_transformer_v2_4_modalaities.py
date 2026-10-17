# -*- coding: utf-8 -*-
"""SWIN_Transformer_V2_4_Modalaities.ipynb

"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Install Libraries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Install monai, nibabel, and torchinfo. Same as UNETR pipeline.
#   MONAI includes a built-in SwinUNETR implementation which uses the
#   Swin Transformer V2 architecture internally, so no extra package
#   is needed beyond what UNETR already required.
# ─────────────────────────────────────────────────────────────────────────────

!pip install monai nibabel torchinfo -q

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='monai')

print("=" * 55)
print("CELL 1 — LIBRARIES INSTALLED")
print("=" * 55)
print("  monai     ✓  Medical imaging framework")
print("  nibabel   ✓  NIfTI file reader")
print("  torchinfo ✓  Model summary printer")
print()
print("  SwinUNETR (SwinV2 backbone) is built into MONAI ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Mount Google Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Mount Google Drive for persistent dataset access and output saving.
#   Identical to UNETR Cell 2.
# ─────────────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

print("=" * 55)
print("CELL 2 — GOOGLE DRIVE MOUNTED")
print("=" * 55)
print("  Mount point : /content/drive/MyDrive")
print("  Status      : Connected ✓")

from IPython.display import display, Javascript

def keep_alive():
    display(Javascript('''
        function click_connect(){
            document.querySelector("#top-toolbar > colab-connect-button")
                    .shadowRoot
                    .querySelector("#connect")
                    .click();
        }
        setInterval(click_connect, 60000);
    '''))

keep_alive()
print("  Keep-alive running ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — All Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Import all required libraries. Key difference from UNETR:
#     SwinUNETR replaces UNETR in the model import.
#     SwinUNETR is MONAI's implementation that uses Swin Transformer V2
#     as its encoder backbone with a UNet-style decoder.
#   All other imports — transforms, metrics, losses — remain identical
#   to the UNETR pipeline ensuring a fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

import os, glob, time, random
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
from monai.networks.nets  import SwinUNETR      # KEY CHANGE from UNETR
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
print("CELL 3 — ENVIRONMENT")
print("=" * 55)
print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {device}")
if torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  GPU RAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  ⚠️  GPU not active — Runtime → T4 GPU → Save")
print("  Seed       : 42 ✓")
print("  SwinUNETR  : imported ✓  (Swin Transformer V2 backbone)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Paths & Global Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define all paths and constants. Key differences from UNETR:
#     MODEL_NAME   = 'SwinUNETR'  — identifies this model in all CSVs
#     SAVE_DIR points to a separate swinunetr/ output folder so UNETR
#     and SwinUNETR outputs never overwrite each other on Drive.
#   All other constants (MAX_EPOCHS, VAL_INTERVAL, CKPT_INTERVAL,
#   DATASET_TYPE) are identical for a fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

BASE     = '/content/drive/MyDrive/Research/Implementation/BRATS_DATASET/'
DATA_DIR = BASE + 'MICCAI_BraTS2020_TrainingData/'
SAVE_DIR = BASE + 'pytorch_outputs/swinunetr/'       # separate from UNETR
CKPT_DIR = SAVE_DIR + 'checkpoints/'
LOG_DIR  = SAVE_DIR + 'logs/'

LOG_TRAIN = LOG_DIR + 'train_log.csv'
LOG_VAL   = LOG_DIR + 'val_log.csv'
LOG_FULL  = LOG_DIR + 'full_log.csv'
LOG_EFF   = LOG_DIR + 'efficiency_log.csv'

MODEL_NAME    = 'SwinUNETR'       # KEY CHANGE: identifies model in CSVs
DATASET_TYPE  = 'BraTS2020'
MAX_EPOCHS    = 100
VAL_INTERVAL  = 4. ## original it was 2
CKPT_INTERVAL = 4 ## Original it was 2

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

cases = sorted(glob.glob(DATA_DIR + 'BraTS20_Training_*'))

print("=" * 55)
print("CELL 4 — PATH VERIFICATION")
print("=" * 55)
print(f"  DATA_DIR exists : {os.path.exists(DATA_DIR)}")
print(f"  Total cases     : {len(cases)}")
print(f"  MODEL_NAME      : {MODEL_NAME}")
print(f"  DATASET_TYPE    : {DATASET_TYPE}")
print(f"  MAX_EPOCHS      : {MAX_EPOCHS}")
print(f"  VAL_INTERVAL    : {VAL_INTERVAL}")
print(f"  CKPT_INTERVAL   : {CKPT_INTERVAL}")
print()
print(f"  SAVE_DIR  : {SAVE_DIR}")
print(f"  CKPT_DIR  : {CKPT_DIR}")
print(f"  LOG_DIR   : {LOG_DIR}")
print()
print("  Output folder separate from UNETR ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Data Dictionaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Scan BraTS 2020 folder and build patient file dictionaries.
#   Identical to UNETR Cell 5 — same dataset, same four modalities,
#   same find_nii logic preferring .nii.gz over .nii.
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
                ['flair', 't1', 't1ce', 't2', 'seg'],
                [flair, t1, t1ce, t2, seg]) if v is None]
            print(f"  ⚠️  Skipped : {name}  missing: {missing}")

    return data_dicts, skipped


all_data, skipped = create_data_dicts(DATA_DIR)

if len(all_data) == 0:
    raise ValueError("No valid cases found. Check DATA_DIR.")

d = all_data[0]
print("=" * 55)
print("CELL 5 — DATA DICTIONARIES CREATED")
print("=" * 55)
print(f"  Valid   : {len(all_data)} cases ✓")
print(f"  Skipped : {len(skipped)} cases")
print()
print("  Sample — first case:")
print(f"    name     : {d['name']}")
print(f"    image[0] : {os.path.basename(d['image'][0])}  ← FLAIR")
print(f"    image[1] : {os.path.basename(d['image'][1])}  ← T1")
print(f"    image[2] : {os.path.basename(d['image'][2])}  ← T1ce")
print(f"    image[3] : {os.path.basename(d['image'][3])}  ← T2")
print(f"    label    : {os.path.basename(d['label'])}  ← Segmentation")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Train / Val Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   80/20 patient-level split with seed=42. Identical to UNETR Cell 6.
#   Using the exact same seed guarantees SwinUNETR trains and validates
#   on the exact same patient cases as UNETR, making the comparison fair.
# ─────────────────────────────────────────────────────────────────────────────

random.seed(42)
shuffled   = all_data.copy()
random.shuffle(shuffled)
split      = int(len(shuffled) * 0.80)
train_data = shuffled[:split]
val_data   = shuffled[split:]

print("=" * 55)
print("CELL 6 — TRAIN / VAL SPLIT")
print("=" * 55)
print(f"  Total  : {len(all_data)}")
print(f"  Train  : {len(train_data)}  (80%)")
print(f"  Val    : {len(val_data)}  (20%)")
print(f"  Seed   : 42  — same split as UNETR for fair comparison ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define preprocessing and augmentation pipelines.
#   Identical to UNETR Cell 7 — same transforms, same crop size (128³),
#   same augmentation probabilities. This ensures any difference in
#   results between UNETR and SwinUNETR is due to the architecture
#   alone and not differences in data preprocessing.
# ─────────────────────────────────────────────────────────────────────────────

class ConvertBraTSLabels(MapTransform):
    """Remap BraTS 2020 label 4 → 3 for contiguous class indices."""
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
    """Unwrap single-element list from RandCropByPosNegLabeld."""
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
print("CELL 7 — TRANSFORMS DEFINED")
print("=" * 55)
print("  Identical to UNETR transforms for fair comparison ✓")
print("  Label remap  : 4 → 3  ✓")
print("  Crop size    : 128³   ✓")
print("  Augmentation : flip×3 + rotate90  ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Datasets + DataLoaders + Sample Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Create datasets and dataloaders. Identical to UNETR Cell 8.
#   Sample verification confirms 4-channel input and correct label values.
# ─────────────────────────────────────────────────────────────────────────────

train_dataset = MonaiDataset(data=train_data, transform=train_transforms)
val_dataset   = MonaiDataset(data=val_data,   transform=val_transforms)

train_loader = MonaiDataLoader(
    train_dataset, batch_size=1,
    shuffle=True, num_workers=2, pin_memory=True)
val_loader = MonaiDataLoader(
    val_dataset, batch_size=1,
    shuffle=False, num_workers=2, pin_memory=True)

print("=" * 55)
print("CELL 8 — DATASETS + DATALOADERS")
print("=" * 55)
print(f"  Train dataset : {len(train_dataset)}")
print(f"  Val   dataset : {len(val_dataset)}")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val   batches : {len(val_loader)}")

print()
print("─" * 55)
print("SAMPLE VERIFICATION")
print("─" * 55)

sample = train_dataset[0]
if isinstance(sample, list):
    sample = sample[0]

img_shape  = sample['image'].shape
lbl_shape  = sample['label'].shape
unique_lbs = torch.unique(sample['label']).tolist()

print(f"  Image shape  : {list(img_shape)}  (C, D, H, W)")
print(f"  Label shape  : {list(lbl_shape)}")
print(f"  Label values : {unique_lbs}")

assert img_shape[0] == 4,  f"Expected 4 channels, got {img_shape[0]}"
assert lbl_shape[0] == 1,  f"Expected 1 label channel, got {lbl_shape[0]}"
assert set(unique_lbs).issubset({0.0, 1.0, 2.0, 3.0}), \
    f"Unexpected label values: {unique_lbs}"

print(f"  ✅ 4 channels confirmed  (FLAIR, T1, T1ce, T2)")
print(f"  ✅ Label remap 4→3 applied")
print(f"  ✅ DataLoaders ready")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Build SwinUNETR Model, Optimiser, Scheduler, Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Instantiate SwinUNETR with Swin Transformer V2 backbone.
#   img_size was removed in newer MONAI versions — the model now infers
#   spatial size dynamically from the input tensor at forward pass time.
#   All other arguments remain the same.
# ─────────────────────────────────────────────────────────────────────────────

# First check which MONAI version is installed so we use the correct signature
import monai
print(f"  MONAI version : {monai.__version__}")

import inspect
swin_args = inspect.signature(SwinUNETR.__init__).parameters
print(f"  SwinUNETR args: {list(swin_args.keys())}")
print()

# Build model with correct signature for current MONAI version
if 'img_size' in swin_args:
    # Older MONAI (< 1.3) — img_size required
    model = SwinUNETR(
        img_size       = (128, 128, 128),
        in_channels    = 4,
        out_channels   = 4,
        depths         = (2, 2, 2, 2),
        num_heads      = (3, 6, 12, 24),
        feature_size   = 48,
        use_checkpoint = True,
        use_v2         = True,
    ).to(device)
else:
    # Newer MONAI (>= 1.3) — img_size removed, size inferred dynamically
    model = SwinUNETR(
        in_channels    = 4,
        out_channels   = 4,
        depths         = (2, 2, 2, 2),
        num_heads      = (3, 6, 12, 24),
        feature_size   = 48,
        use_checkpoint = True,
        use_v2         = True,
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

print("=" * 55)
print("CELL 9 — MODEL READY — SwinUNETR (Swin V2)")
print("=" * 55)
print(f"  Architecture     : SwinUNETR (Swin Transformer V2)")
print(f"  Dataset          : {DATASET_TYPE}")
print(f"  MONAI version    : {monai.__version__}")
print(f"  img_size arg     : {'used' if 'img_size' in swin_args else 'removed — inferred dynamically'}")
print()
print("  Swin V2 Encoder:")
print(f"    In channels    : 4   (FLAIR, T1, T1ce, T2)")
print(f"    Feature size   : 48  (base dimension)")
print(f"    Depths         : (2, 2, 2, 2)  blocks per stage")
print(f"    Num heads      : (3, 6, 12, 24)  per stage")
print(f"    use_v2         : True  ← Scaled cosine attention")
print(f"                            Log-spaced position bias")
print(f"                            Pre-normalisation")
print(f"    Grad ckpt      : True  ← saves GPU memory")
print()
print("  Decoder:")
print(f"    Out channels   : 4   (BG, NCR, Edema, ET)")
print()
print("  Training:")
print(f"    Optimizer      : Adam  lr=1e-4  wd=1e-5")
print(f"    Scheduler      : CosineAnnealingLR  T_max={MAX_EPOCHS}")
print(f"    Loss           : DiceCELoss  0.5+0.5")
print()
print("  Parameters:")
print(f"    Total          : {total_params:,}")
print(f"    Trainable      : {trainable_params:,}")
print(f"    Non-trainable  : {non_trainable:,}")
print(f"    Model size     : {total_params * 4 / 1024 / 1024:.2f} MB")
print(f"    Device         : {device}")
print()
print("  Model ready ✓")
print("  Ready for Cell 10 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 — Define All Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Instantiate all evaluation metric objects used during validation.
#   Completely identical to UNETR Cell 10 — same metrics, same config,
#   same fixes. This guarantees metric computation is identical across
#   both models for a fair comparison.
#
#   include_background=True for dice_wt, dice_tc, dice_et because
#   sub-region tensors are single-channel (1,1,D,H,W). Using False
#   would treat the only channel as background and return NaN silently.
# ─────────────────────────────────────────────────────────────────────────────

dice_metric = DiceMetric(
    include_background = False,
    reduction          = 'mean_batch',
    get_not_nans       = True,
)
iou_metric = MeanIoU(
    include_background = False,
    reduction          = 'mean_batch',
    get_not_nans       = True,
)
confusion_metric = ConfusionMatrixMetric(
    include_background = False,
    metric_name        = ['sensitivity', 'specificity', 'precision'],
    reduction          = 'mean_batch',
    get_not_nans       = True,
)

# include_background=True — sub-region tensors are single-channel binary maps
dice_wt = DiceMetric(
    include_background = True,
    reduction          = 'mean',
    get_not_nans       = True,
)
dice_tc = DiceMetric(
    include_background = True,
    reduction          = 'mean',
    get_not_nans       = True,
)
dice_et = DiceMetric(
    include_background = True,
    reduction          = 'mean',
    get_not_nans       = True,
)

post_pred  = AsDiscrete(argmax=True, to_onehot=4)
post_label = AsDiscrete(to_onehot=4)

print("=" * 55)
print("CELL 10 — METRICS DEFINED")
print("=" * 55)
print(f"  Dataset : {DATASET_TYPE}")
print(f"  Model   : {MODEL_NAME}")
print()
print("  Primary metrics  (include_background=False):")
print("    DiceMetric           ✓  main segmentation metric")
print("    MeanIoU              ✓  Jaccard index")
print("    ConfusionMatrix      ✓  sensitivity, specificity, precision")
print()
print("  BraTS sub-region  (include_background=True):")
print("    Dice WT  ✓  ch1+ch2+ch3  (whole tumour)")
print("    Dice TC  ✓  ch1+ch3      (tumour core)")
print("    Dice ET  ✓  ch3          (enhancing tumour)")
print()
print("  HD95 : skipped during training — final eval cell only")
print("         val_hd95 = float('nan') in logs")
print()
print("  post_pred   ✓  argmax → one-hot(4)")
print("  post_label  ✓  integer → one-hot(4)")
print()
print("  Identical to UNETR Cell 10 for fair comparison ✓")
print("  All metrics defined ✓")
print("  Ready for Cell 11 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — Measure Efficiency Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Measure SwinUNETR efficiency statistics for the paper comparison table.
#   Identical measurement methodology to UNETR Cell 11:
#     3 warmup runs + 10 timed runs using CUDA events.
#     GPU cache cleared after measurement.
#   Results will differ from UNETR because SwinUNETR has a different
#   parameter count, memory footprint and computational graph.
#   use_checkpoint=True in SwinUNETR means activations are recomputed
#   during backward pass — this lowers peak GPU memory during training
#   but does not affect inference memory measured here (no backward pass).
# ─────────────────────────────────────────────────────────────────────────────

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
        gpu_memory_MB = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.time()
                model(dummy)
                timings.append((time.time() - t0) * 1000)
        inference_ms  = np.mean(timings)
        gpu_memory_MB = 0.0

    del dummy
    torch.cuda.empty_cache()

    return {
        'total_params'  : total_params,
        'params_M'      : round(total_params / 1e6,             4),
        'model_size_MB' : round(total_params * 4 / 1024 / 1024, 2),
        'inference_ms'  : round(inference_ms,                    2),
        'gpu_memory_MB' : round(gpu_memory_MB,                   2),
    }


print("=" * 55)
print("CELL 11 — EFFICIENCY METRICS")
print("=" * 55)
print(f"  Model      : {MODEL_NAME}")
print(f"  Dataset    : {DATASET_TYPE}")
print(f"  Device     : {device}")
print(f"  Input shape: (1, 4, 128, 128, 128)")
print(f"  Warmup     : 3 runs")
print(f"  Timed      : 10 runs")
print()
print("  Measuring...")
print()

eff = measure_efficiency(model, device)

total_params     = eff['total_params']
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
non_trainable    = total_params - trainable_params

print(f"  Parameters     : {total_params:,}")
print(f"  Parameters (M) : {eff['params_M']:.4f} M")
print(f"  Trainable      : {trainable_params:,}")
print(f"  Non-trainable  : {non_trainable:,}")
print(f"  Model size     : {eff['model_size_MB']:.2f} MB")
print(f"  Inference time : {eff['inference_ms']:.2f} ms / sample")
print(f"  GPU memory     : {eff['gpu_memory_MB']:.2f} MB")
print()
print("  GPU cache cleared after measurement ✓")
print("  Efficiency values ready for paper table ✓")
print("  Ready for Cell 12 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — Model Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Print layer-by-layer summary of SwinUNETR using torchinfo.
#   Unlike UNETR which shows uniform ViT layers, SwinUNETR summary
#   shows the 4-stage hierarchical encoder with progressively halving
#   spatial dimensions and doubling feature channels at each stage,
#   followed by the symmetric UNet-style decoder with skip connections.
#
#   Expected shape flow:
#     Input    : (1,   4, 128, 128, 128)
#     Stage 1  : (1,  48,  64,  64,  64)  patch partition + embed
#     Stage 2  : (1,  96,  32,  32,  32)  patch merging
#     Stage 3  : (1, 192,  16,  16,  16)  patch merging
#     Stage 4  : (1, 384,   8,   8,   8)  patch merging
#     Decoder  : (1,   4, 128, 128, 128)  progressive upsample
#
#   Dummy input deleted and GPU cache cleared immediately after
#   so full memory is available for training in Cell 15.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("CELL 12 — MODEL SUMMARY")
print("=" * 55)
print(f"  Model      : {MODEL_NAME}")
print(f"  Dataset    : {DATASET_TYPE}")
print(f"  Device     : {device}")
print(f"  Params     : {eff['params_M']:.4f} M")
print(f"  Size       : {eff['model_size_MB']:.2f} MB")
print(f"  Inference  : {eff['inference_ms']:.2f} ms / sample")
print(f"  GPU memory : {eff['gpu_memory_MB']:.2f} MB")
print()
print("  Expected stage-wise shape flow:")
print("    Input    : (1,   4, 128, 128, 128)")
print("    Stage 1  : (1,  48,  64,  64,  64)  patch partition")
print("    Stage 2  : (1,  96,  32,  32,  32)  patch merging")
print("    Stage 3  : (1, 192,  16,  16,  16)  patch merging")
print("    Stage 4  : (1, 384,   8,   8,   8)  patch merging")
print("    Output   : (1,   4, 128, 128, 128)  decoder upsample")
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
torch.cuda.empty_cache()

print()
print("=" * 55)
print("SUMMARY VERIFICATION")
print("=" * 55)
print(f"  Expected input  : (1, 4, 128, 128, 128)")
print(f"  Expected output : (1, 4, 128, 128, 128)")
print(f"  Expected params : {eff['params_M']:.4f} M")
print(f"  use_v2          : True  ← Swin V2 confirmed")
print(f"  use_checkpoint  : True  ← gradient checkpointing confirmed")
print()
print("  Model summary printed ✓")
print("  GPU cache cleared     ✓")
print("  Ready for Cell 13     ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — Training & Validation Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define all functions called inside the training loop.
#   Completely identical to UNETR Cell 13 — same four functions,
#   same arguments, same logic. This guarantees training and evaluation
#   procedure is controlled so any performance difference between
#   UNETR and SwinUNETR reflects architecture alone.
#
#   Note on use_checkpoint=True in SwinUNETR:
#     Gradient checkpointing means activations are NOT stored during
#     the forward pass. Instead they are recomputed during backward.
#     This reduces peak GPU memory at the cost of ~20% slower backward.
#     train_one_epoch() does not need any changes — PyTorch handles
#     this transparently through the model's forward() method.
# ─────────────────────────────────────────────────────────────────────────────

def extract_subregions(pred_onehot, label_onehot):
    """
    Input  : (4, D, H, W) one-hot tensors — no batch dimension.
    Output : 6 tensors of shape (1, 1, D, H, W)
               pred_wt/label_wt, pred_tc/label_tc, pred_et/label_et
    """
    assert pred_onehot.ndim == 4, \
        f"Expected 4-D tensor (C,D,H,W), got ndim={pred_onehot.ndim}"
    assert pred_onehot.shape[0] == 4, \
        f"Expected 4 one-hot channels, got {pred_onehot.shape[0]}"

    pred_wt  = (pred_onehot[1] + pred_onehot[2]
                + pred_onehot[3]).clamp(0, 1)
    label_wt = (label_onehot[1] + label_onehot[2]
                + label_onehot[3]).clamp(0, 1)
    pred_tc  = (pred_onehot[1] + pred_onehot[3]).clamp(0, 1)
    label_tc = (label_onehot[1] + label_onehot[3]).clamp(0, 1)
    pred_et  = pred_onehot[3]
    label_et = label_onehot[3]

    def wrap(t):
        return t.unsqueeze(0).unsqueeze(0)   # → (1, 1, D, H, W)

    return (wrap(pred_wt),  wrap(label_wt),
            wrap(pred_tc),  wrap(label_tc),
            wrap(pred_et),  wrap(label_et))


def extract_conf(result, idx):
    """
    Safely extract a float scalar from ConfusionMatrixMetric.aggregate()
    regardless of MONAI version return format.
    """
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
    """
    One full training epoch.
    Returns (mean_loss, epoch_time_minutes).
    Gradient clipping at max_norm=1.0 — same as UNETR.
    use_checkpoint=True handled transparently by PyTorch.
    """
    model.train()
    epoch_loss  = 0.0
    steps       = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(loader):
        steps  += 1
        images  = batch['image'].to(device)   # (1, 4, 128, 128, 128)
        labels  = batch['label'].to(device)   # (1, 1, 128, 128, 128)

        optimizer.zero_grad()
        outputs = model(images)               # (1, 4, 128, 128, 128)
        loss    = loss_fn(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    [{batch_idx+1:>4}/{len(loader)}]"
                  f"  Loss: {loss.item():.4f}")

    epoch_time_min = (time.time() - epoch_start) / 60.0
    return epoch_loss / steps, epoch_time_min


def validate(model, loader, loss_fn,
             dice_metric, iou_metric,
             confusion_metric,
             dice_wt, dice_tc, dice_et,
             post_pred, post_label, device):
    """
    Full validation on complete volumes via sliding window inference.
    overlap=0.5, mode='gaussian' — identical to UNETR validation.
    val_hd95=float('nan') — computed in final eval cell only.
    All metric accumulators reset after aggregation.
    """
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
                (p_wt, l_wt,
                 p_tc, l_tc,
                 p_et, l_et) = extract_subregions(pred_oh, label_oh)
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
        val_sens = conf_result[..., 0].mean().item()
        val_spec = conf_result[..., 1].mean().item()
        val_prec = conf_result[..., 2].mean().item()

    val_wt = get_val(dice_wt.aggregate())
    val_tc = get_val(dice_tc.aggregate())
    val_et = get_val(dice_et.aggregate())

    for m in [dice_metric, iou_metric, confusion_metric,
              dice_wt, dice_tc, dice_et]:
        m.reset()

    return {
        'val_loss' : val_loss / steps,
        'val_dice' : val_dice,
        'val_iou'  : val_iou,
        'val_hd95' : float('nan'),
        'val_sens' : val_sens,
        'val_spec' : val_spec,
        'val_prec' : val_prec,
        'val_wt'   : val_wt,
        'val_tc'   : val_tc,
        'val_et'   : val_et,
    }


# ── Sanity test ───────────────────────────────────────────────────────────────
print("=" * 55)
print("CELL 13 — FUNCTIONS DEFINED + SANITY TEST")
print("=" * 55)

dummy_pred  = [torch.zeros(4, 32, 32, 32)]
dummy_label = [torch.zeros(4, 32, 32, 32)]
dummy_pred[0][0,  ...] = 1
dummy_label[0][0, ...] = 1

confusion_metric(y_pred=dummy_pred, y=dummy_label)
test_result = confusion_metric.aggregate()
confusion_metric.reset()

print(f"  ConfusionMatrix return type : {type(test_result)}")
print(f"  Sensitivity : {extract_conf(test_result, 0):.4f}")
print(f"  Specificity : {extract_conf(test_result, 1):.4f}")
print(f"  Precision   : {extract_conf(test_result, 2):.4f}")
print()
print("  Function summary:")
print("    extract_subregions() ✓  ndim + channel guard")
print("    extract_conf()       ✓  multi-format safe")
print("    train_one_epoch()    ✓  grad clip max_norm=1.0")
print("                            use_checkpoint handled by PyTorch")
print("    validate()           ✓  overlap=0.5  gaussian mode")
print("                            val_hd95=float('nan')")
print("                            all metrics reset after aggregation")
print()
print("  Identical to UNETR Cell 13 for fair comparison ✓")
print("  All training functions ready ✓")
print("  Ready for Cell 14 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14 — Define save_logs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define save_logs() to write four CSV files after every validation epoch.
#   Functionally identical to UNETR Cell 14.
#   MODEL_NAME = 'SwinUNETR' flows in from Cell 4 automatically into
#   the 'model' column of efficiency_log.csv, so when UNETR and SwinUNETR
#   efficiency logs are stacked in pandas for comparison, each row is
#   clearly labelled by model name.
#
#   Four CSVs written to swinunetr/logs/ (separate from UNETR logs):
#     train_log.csv      — epoch, train_loss, lr, train_time_min
#     val_log.csv        — all 10 val metrics at val epochs
#     full_log.csv       — merged train+val rows aligned by epoch
#     efficiency_log.csv — static efficiency + per-epoch time + MEAN row
# ─────────────────────────────────────────────────────────────────────────────

def save_logs(history, log_train, log_val,
              log_full, log_eff, eff_static):

    # train_log.csv
    pd.DataFrame({
        'epoch'          : history['epoch'],
        'train_loss'     : history['train_loss'],
        'lr'             : history['lr'],
        'train_time_min' : history['train_time_min'],
    }).to_csv(log_train, index=False)

    # val_log.csv
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

    # full_log.csv — merge train and val by epoch number
    val_lookup = {ve: i for i, ve in enumerate(history['val_epoch'])}
    full_rows  = []
    for i, ep in enumerate(history['epoch']):
        row = {
            'epoch'          : ep,
            'train_loss'     : history['train_loss'][i],
            'lr'             : history['lr'][i],
            'train_time_min' : history['train_time_min'][i],
            'val_loss' : None, 'val_dice' : None,
            'val_iou'  : None, 'val_hd95' : None,
            'val_sens' : None, 'val_spec' : None,
            'val_prec' : None, 'val_wt'   : None,
            'val_tc'   : None, 'val_et'   : None,
        }
        if ep in val_lookup:
            vi = val_lookup[ep]
            row.update({
                'val_loss' : history['val_loss'][vi],
                'val_dice' : history['val_dice'][vi],
                'val_iou'  : history['val_iou'][vi],
                'val_hd95' : history['val_hd95'][vi],
                'val_sens' : history['val_sens'][vi],
                'val_spec' : history['val_spec'][vi],
                'val_prec' : history['val_prec'][vi],
                'val_wt'   : history['val_wt'][vi],
                'val_tc'   : history['val_tc'][vi],
                'val_et'   : history['val_et'][vi],
            })
        full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(log_full, index=False)

    # efficiency_log.csv
    # mean_time computed before comprehension to avoid variable leak
    mean_time = round(np.mean(history['train_time_min']), 4)

    eff_rows = [{
        'model'          : MODEL_NAME,        # 'SwinUNETR'
        'dataset'        : DATASET_TYPE,      # 'BraTS2020'
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

    pd.concat(
        [pd.DataFrame(eff_rows), summary_row],
        ignore_index=True
    ).to_csv(log_eff, index=False)


print("=" * 55)
print("CELL 14 — SAVE_LOGS DEFINED")
print("=" * 55)
print(f"  model column    : {MODEL_NAME}")
print(f"  dataset column  : {DATASET_TYPE}")
print()
print("  4 CSV outputs per validation epoch:")
print(f"    {LOG_TRAIN}")
print(f"    {LOG_VAL}")
print(f"    {LOG_FULL}")
print(f"    {LOG_EFF}")
print()
print("  val_epoch list for alignment  ✓")
print("  mean_time before loop         ✓")
print("  model = SwinUNETR             ✓  stackable with UNETR CSVs")
print("  save_logs defined ✓")
print("  Ready for Cell 15 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15 — Training Loop (25 Epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Run the main SwinUNETR training loop for MAX_EPOCHS epochs.
#   Structure is word for word identical to UNETR Cell 15.
#   Five steps per epoch:
#     1. Train one epoch  → mean loss + wall-clock time
#     2. Step cosine LR scheduler
#     3. Record train metrics in history dict
#     4. Every VAL_INTERVAL epochs:
#          • Run full validation → 10 metrics
#          • Write all four CSVs to Drive
#          • Save best_model.pth if Dice improved
#     5. Every CKPT_INTERVAL epochs:
#          • Save numbered checkpoint for recovery
#
#   Checkpoints saved to swinunetr/checkpoints/ on Google Drive,
#   separate from UNETR checkpoints so they never overwrite each other.
#   model_name='SwinUNETR' in each checkpoint confirms correct
#   architecture when loading for inference or resuming training.
# ─────────────────────────────────────────────────────────────────────────────

history = {
    'epoch'          : [],
    'train_loss'     : [],
    'lr'             : [],
    'train_time_min' : [],
    'val_epoch'      : [],
    'val_loss'       : [],
    'val_dice'       : [],
    'val_iou'        : [],
    'val_hd95'       : [],
    'val_sens'       : [],
    'val_spec'       : [],
    'val_prec'       : [],
    'val_wt'         : [],
    'val_tc'         : [],
    'val_et'         : [],
}

best_dice  = 0.0
best_epoch = 0

print("=" * 55)
print(f"CELL 15 — STARTING TRAINING")
print("=" * 55)
print(f"  Model          : {MODEL_NAME}")
print(f"  Dataset        : {DATASET_TYPE}")
print(f"  Device         : {device}")
print()
print(f"  Train cases    : {len(train_dataset)}")
print(f"  Val   cases    : {len(val_dataset)}")
print(f"  Train batches  : {len(train_loader)}")
print(f"  Val   batches  : {len(val_loader)}")
print()
print(f"  Max epochs     : {MAX_EPOCHS}")
print(f"  Val interval   : every {VAL_INTERVAL} epochs")
print(f"  Ckpt interval  : every {CKPT_INTERVAL} epochs")
print(f"  Batch size     : 1")
print(f"  sw_batch_size  : 4")
print()
print(f"  SwinV2 config:")
print(f"    use_v2         : True")
print(f"    use_checkpoint : True  (gradient checkpointing)")
print(f"    depths         : (2, 2, 2, 2)")
print(f"    num_heads      : (3, 6, 12, 24)")
print(f"    feature_size   : 48")
print()
print(f"  Params         : {eff['params_M']:.2f} M")
print(f"  Model size     : {eff['model_size_MB']:.1f} MB")
print(f"  Inference      : {eff['inference_ms']:.1f} ms / sample")
print(f"  GPU memory     : {eff['gpu_memory_MB']:.1f} MB")
print()
print(f"  Checkpoints  → {CKPT_DIR}")
print(f"  Logs         → {LOG_DIR}")
print("=" * 55)

# ── Main training loop ────────────────────────────────────────────────────────
for epoch in range(1, MAX_EPOCHS + 1):

    print(f"\n{'=' * 55}")
    print(f"  Epoch {epoch}/{MAX_EPOCHS}"
          f"  [{MODEL_NAME} — {DATASET_TYPE}]")
    print(f"  {'-' * 51}")

    # 1. Train one epoch
    train_loss, epoch_time = train_one_epoch(
        model, train_loader,
        optimizer, loss_fn,
        device, epoch)

    # 2. Step LR scheduler
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # 3. Record train metrics
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['lr'].append(lr)
    history['train_time_min'].append(round(epoch_time, 4))

    print(f"\n  Train Loss  : {train_loss:.4f}")
    print(f"  LR          : {lr:.7f}")
    print(f"  Epoch time  : {epoch_time:.2f} min")

    # 4. Validation
    if epoch % VAL_INTERVAL == 0:

        print(f"\n  Running validation...")

        metrics = validate(
            model, val_loader, loss_fn,
            dice_metric, iou_metric,
            confusion_metric,
            dice_wt, dice_tc, dice_et,
            post_pred, post_label, device)

        history['val_epoch'].append(epoch)
        for k in ['val_loss', 'val_dice', 'val_iou', 'val_hd95',
                  'val_sens', 'val_spec', 'val_prec',
                  'val_wt',   'val_tc',   'val_et']:
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
        print(f"\n  ── BraTS Sub-region Metrics ─────────────")
        print(f"  Dice WT     : {metrics['val_wt']:.4f}"
              f"  (NCR + Edema + ET)")
        print(f"  Dice TC     : {metrics['val_tc']:.4f}"
              f"  (NCR + ET)")
        print(f"  Dice ET     : {metrics['val_et']:.4f}"
              f"  (ET only)")
        print(f"\n  ── Efficiency ───────────────────────────")
        print(f"  Avg epoch time : "
              f"{np.mean(history['train_time_min']):.2f} min")
        print(f"  Total so far   : "
              f"{sum(history['train_time_min']):.2f} min")

        # Write all four CSVs
        save_logs(history,
                  LOG_TRAIN, LOG_VAL,
                  LOG_FULL,  LOG_EFF, eff)
        print(f"\n  CSVs saved to Drive ✓")

        # Save best model if Dice improved
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
                'model_name' : MODEL_NAME,    # 'SwinUNETR'
                'dataset'    : DATASET_TYPE,  # 'BraTS2020'
            }, CKPT_DIR + 'best_model.pth')
            print(f"  ✓ Best model saved!")
            print(f"    Dice  = {best_dice:.4f}")
            print(f"    Epoch = {best_epoch}")

    # 5. Periodic checkpoint
    if epoch % CKPT_INTERVAL == 0:
        ckpt_path = CKPT_DIR + f'epoch_{epoch:03d}.pth'
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_dice'  : best_dice,
            'efficiency' : eff,
            'history'    : history,
            'model_name' : MODEL_NAME,
            'dataset'    : DATASET_TYPE,
        }, ckpt_path)
        print(f"  Checkpoint saved → epoch_{epoch:03d}.pth ✓")

# Final CSV save — ensures last epoch always written
save_logs(history, LOG_TRAIN, LOG_VAL, LOG_FULL, LOG_EFF, eff)

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"TRAINING COMPLETE")
print("=" * 55)
print(f"  Model         : {MODEL_NAME}")
print(f"  Dataset       : {DATASET_TYPE}")
print()
print(f"  Best Dice     : {best_dice:.4f}  (Epoch {best_epoch})")
print(f"  Last Dice     : {history['val_dice'][-1]:.4f}")
print(f"  Last IoU      : {history['val_iou'][-1]:.4f}")
print(f"  Last Sens     : {history['val_sens'][-1]:.4f}")
print(f"  Last Spec     : {history['val_spec'][-1]:.4f}")
print(f"  Last Prec     : {history['val_prec'][-1]:.4f}")
print(f"  Last WT       : {history['val_wt'][-1]:.4f}")
print(f"  Last TC       : {history['val_tc'][-1]:.4f}")
print(f"  Last ET       : {history['val_et'][-1]:.4f}")
print()
print(f"  Efficiency:")
print(f"  Params        : {eff['params_M']:.4f} M")
print(f"  Model size    : {eff['model_size_MB']:.2f} MB")
print(f"  Inference     : {eff['inference_ms']:.2f} ms")
print(f"  GPU memory    : {eff['gpu_memory_MB']:.2f} MB")
print(f"  Avg / epoch   : {np.mean(history['train_time_min']):.2f} min")
print(f"  Total time    : {sum(history['train_time_min']):.2f} min")
print()
print(f"  Saved to:")
print(f"    {CKPT_DIR}best_model.pth")
print(f"    {LOG_TRAIN}")
print(f"    {LOG_VAL}")
print(f"    {LOG_FULL}")
print(f"    {LOG_EFF}")
print()
print("  Next → Run Final Evaluation Cell ✓")



#### Resume training from epochs 19 again after disconnect

"""STEP  1  →  Run Cell 1    install libraries
STEP  2  →  Run Cell 2    mount Drive
STEP  3  →  Run Keep Alive
STEP  4  →  Run Cell 3    imports
STEP  5  →  Run Cell 4    paths and constants
STEP  6  →  Run Cells 5, 6, 7, 8
STEP  7  →  Run Cell 9    build model
STEP  8  →  Run Cells 10, 11, 13, 14
STEP  9  →  Run Checkpoint Status cell
STEP 10  →  Run Resume cell  ← auto finds latest checkpoint
STEP 11  →  Run Cell 15   training continues from START_EPOCH
"""

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

import gc

all_ckpts = sorted(glob.glob(CKPT_DIR + 'epoch_*.pth'))

if len(all_ckpts) > 0:
    latest_ckpt = all_ckpts[-1]

    checkpoint = torch.load(
        latest_ckpt,
        map_location = device,
        weights_only = False    # PyTorch 2.6 fix
    )

    assert checkpoint['model_name'] == 'SwinUNETR'
    assert checkpoint['dataset']    == 'BraTS2020'

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    history     = checkpoint['history']
    best_dice   = checkpoint['best_dice']
    START_EPOCH = checkpoint['epoch'] + 1

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max      = MAX_EPOCHS,
        eta_min    = 1e-6,
        last_epoch = checkpoint['epoch']
    )

    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 55)
    print("RESUMED FROM CHECKPOINT")
    print("=" * 55)
    print(f"  Checkpoint   : {os.path.basename(latest_ckpt)}")
    print(f"  Resumed from : epoch {checkpoint['epoch']}")
    print(f"  Starting at  : epoch {START_EPOCH}")
    print(f"  Best Dice    : {best_dice:.4f}")
    print(f"  History rows : {len(history['epoch'])} epochs")
    print(f"  LR           : {optimizer.param_groups[0]['lr']:.7f}")
    print()
    print("  model weights     ✓")
    print("  optimiser state   ✓")
    print("  history dict      ✓")
    print("  scheduler         ✓")
    print("  GPU cache cleared ✓")
else:
    START_EPOCH = 1
    best_dice   = 0.0
    print("  No checkpoint found — starting from epoch 1")

import gc

best_epoch = 0

print("=" * 55)
print(f"RESUMING FROM EPOCH {START_EPOCH}")
print("=" * 55)
print(f"  Model         : {MODEL_NAME}")
print(f"  Dataset       : {DATASET_TYPE}")
print(f"  Start epoch   : {START_EPOCH}")
print(f"  Max epochs    : {MAX_EPOCHS}")
print(f"  Best Dice     : {best_dice:.4f}")
print(f"  Val interval  : every {VAL_INTERVAL} epochs")
print(f"  Ckpt interval : every {CKPT_INTERVAL} epochs")
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

    print(f"\n  Train Loss : {train_loss:.4f}")
    print(f"  LR         : {lr:.7f}")
    print(f"  Epoch time : {epoch_time:.2f} min")

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
print(f"  Total time : {sum(history['train_time_min']):.2f} min")
print("\n  Next → Run Final Evaluation Cell ✓")