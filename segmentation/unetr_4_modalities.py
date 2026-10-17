# -*- coding: utf-8 -*-
"""UNETR_4_Modalities.ipynb


"""

#### UNETR impelemtation on BRATS 2020 Data. Use all modalities

#### Complete code with updation

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Install Libraries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Install the three external libraries not available by default in Colab:
#     monai     — 3-D medical imaging transforms, losses, and metrics.
#     nibabel   — Reads NIfTI (.nii / .nii.gz) brain scan files.
#     torchinfo — Prints a layer-by-layer model parameter summary.
#   MONAI UserWarnings are suppressed to keep output readable.
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Mount Google Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Mount the user's Google Drive so that:
#     • BraTS 2020 dataset folders are accessible at /content/drive/MyDrive/...
#     • Trained checkpoints and CSV logs are saved persistently across sessions.
#   Without mounting, all files are lost when the Colab runtime disconnects.
# ─────────────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

print("=" * 55)
print("CELL 2 — GOOGLE DRIVE MOUNTED")
print("=" * 55)
print("  Mount point : /content/drive/MyDrive")
print("  Status      : Connected ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — All Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Centralise every import in one cell so that:
#     
#   Device is selected automatically: GPU if available, else CPU.
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
    SelectItemsd,                   # BUG-1 FIX: unwrap crop list
)
from monai.data import (
    Dataset    as MonaiDataset,
    DataLoader as MonaiDataLoader,
)
from monai.networks.nets  import UNETR
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

# Device
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
print("  Seed     : 42 ✓")
print("  Imports  : All loaded ✓")
print("  SelectItemsd imported ✓  (BUG-1 fix)")   # closing ) was missing here

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Paths & Global Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define every file-system path and training hyper-parameter constant in one
#   place so that any change (dataset location, epoch count, intervals) only
#   needs to be made here and propagates automatically to all later cells.
#   Output directories are created if they do not already exist.
#   MODEL_NAME and DATASET_TYPE are used as column values in the efficiency CSV
#   
# ─────────────────────────────────────────────────────────────────────────────

BASE     = '/content/drive/MyDrive/Research/Implementation/BRATS_DATASET/'
DATA_DIR = BASE + 'MICCAI_BraTS2020_TrainingData/'
SAVE_DIR = BASE + 'pytorch_outputs/unetr/'
CKPT_DIR = SAVE_DIR + 'checkpoints/'
LOG_DIR  = SAVE_DIR + 'logs/'

LOG_TRAIN = LOG_DIR + 'train_log.csv'
LOG_VAL   = LOG_DIR + 'val_log.csv'
LOG_FULL  = LOG_DIR + 'full_log.csv'
LOG_EFF   = LOG_DIR + 'efficiency_log.csv'

MODEL_NAME    = 'UNETR'
DATASET_TYPE  = 'BraTS2020'
MAX_EPOCHS    = 50
VAL_INTERVAL  = 4
CKPT_INTERVAL = 4

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
print(f"  DATA_DIR  : {DATA_DIR}")
print(f"  CKPT_DIR  : {CKPT_DIR}")
print(f"  LOG_DIR   : {LOG_DIR}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Data Dictionaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Scan the BraTS 2020 training folder and build a list of dictionaries, one
#   per patient case, each containing paths to all four MRI modalities and the
#   segmentation label.  Cases with any missing file are skipped with a warning.
#
# ─────────────────────────────────────────────────────────────────────────────

def find_nii(path):
    """
    Return .nii.gz path if it exists, else .nii, else None.
    Prefers .nii.gz because BraTS 2020 (Kaggle) ships compressed files.
    FIX: original code preferred .nii, causing inconsistent file selection
    when both formats were present in the same directory.
    """
    if os.path.exists(path + '.gz'):   # prefer compressed (BraTS canonical)
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
                'image' : [flair, t1, t1ce, t2],   # 4 modalities
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
print()
print("  find_nii prefers .nii.gz  ✓  (file-order fix)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Train / Val Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Randomly shuffle all patient cases with a fixed seed (42) and split them
#   80% for training and 20% for validation.
#   Using a fixed seed guarantees the same split is reproduced every time the
#   notebook is re-run, making results comparable across experiments.
#   The split is done at the patient level (not slice level) to prevent data
#   leakage between train and val sets.
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
print(f"  Seed   : 42 (fixed — patient-level split)")
print()
print("  First 3 train cases:")
for d in train_data[:3]:
    print(f"    {d['name']}")
print("  First 3 val cases:")
for d in val_data[:3]:
    print(f"    {d['name']}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from monai.transforms import Lambda

class ConvertBraTSLabels(MapTransform):
    """
    Remap BraTS 2020 label values to contiguous indices:
      0 → 0  Background
      1 → 1  NCR / NET
      2 → 2  Edema
      4 → 3  Enhancing Tumor
    Works on both numpy arrays (pre-EnsureTyped) and torch tensors.
    """
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
    """
    BUG-1 FIX: RandCropByPosNegLabeld with num_samples=1 returns a list
    containing one dict. This lambda extracts that dict so the DataLoader
    collator receives a plain dict instead of a list-of-dicts, which would
    crash batch collation on the first training iteration.
    """
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
    Lambda(unwrap_crop_list),           # BUG-1 FIX: unwrap list → plain dict
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
print("  Label remap  : 4 → 3  (BraTS 2020 ET fix)  ✓")
print()
print("  Train transforms:")
print("    LoadImaged              ✓")
print("    EnsureChannelFirstd     ✓")
print("    ConvertBraTSLabels      ✓  4→3 remap")
print("    NormalizeIntensityd     ✓  per-modality nonzero")
print("    RandCropByPosNegLabeld  ✓  pos=1 neg=1 128³")
print("    Lambda(unwrap_crop_list)✓  BUG-1 FIX — unwrap list")
print("    RandFlip ×3             ✓  prob=0.5")
print("    RandRotate90            ✓  prob=0.5")
print("    EnsureTyped             ✓")
print()
print("  Val transforms:")
print("    LoadImaged, EnsureChannelFirstd, ConvertBraTSLabels")
print("    NormalizeIntensityd, EnsureTyped  ✓  no augmentation")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Datasets + DataLoaders + Sample Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Wrap the data dictionaries in MONAI Dataset objects (which apply transforms
#   lazily on __getitem__) and create DataLoader objects for batching.
#   A sample verification block runs immediately to confirm:
#     • Image has 4 channels  (FLAIR, T1, T1ce, T2)
#     • Label has 1 channel
#     • Label values are a subset of {0, 1, 2, 3}  (remap 4→3 applied)
#     • Image values are normalised (nonzero voxels only, per channel)
#   Catching shape or value errors here is far cheaper than discovering them
#   mid-training after hours of computation.
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
print(f"  Batch size    : 1")
print(f"  num_workers   : 2")
print(f"  pin_memory    : True")

# ── Sample verification ───────────────────────────────────────────────────────
print()
print("─" * 55)
print("SAMPLE VERIFICATION")
print("─" * 55)

sample = train_dataset[0]
# After Lambda(unwrap_crop_list) the result is a plain dict.
# Guard retained for safety in case transform order changes.
if isinstance(sample, list):
    sample = sample[0]

img_shape  = sample['image'].shape
lbl_shape  = sample['label'].shape
unique_lbs = torch.unique(sample['label']).tolist()

print(f"  Image shape   : {list(img_shape)}  (C, D, H, W)")
print(f"    channels    : {img_shape[0]}  (FLAIR, T1, T1ce, T2)")
print(f"    spatial     : {img_shape[1]}×{img_shape[2]}×{img_shape[3]}")
print(f"  Label shape   : {list(lbl_shape)}")
print(f"  Image dtype   : {sample['image'].dtype}")
print(f"  Label dtype   : {sample['label'].dtype}")
print(f"  Image range   : {sample['image'].min():.3f} → {sample['image'].max():.3f}")
print(f"  Label values  : {unique_lbs}")
print()

assert img_shape[0] == 4, \
    f"Expected 4 image channels, got {img_shape[0]}"
assert lbl_shape[0] == 1, \
    f"Expected 1 label channel, got {lbl_shape[0]}"
assert set(unique_lbs).issubset({0.0, 1.0, 2.0, 3.0}), \
    f"Unexpected label values: {unique_lbs}"
assert max(unique_lbs) <= 3.0, \
    f"Label out of range: max={max(unique_lbs)}"

print(f"  ✅ 4 channels confirmed  (FLAIR, T1, T1ce, T2)")
print(f"  ✅ Label remap 4→3 applied  max={max(unique_lbs)}")
print(f"  ✅ DataLoaders ready")
print()
print("=" * 55)
print("DATALOADER SUMMARY")
print("=" * 55)
print(f"  ✅ Dataset      : {DATASET_TYPE}")
print(f"  ✅ Train cases  : {len(train_dataset)}")
print(f"  ✅ Val   cases  : {len(val_dataset)}")
print(f"  ✅ Image shape  : {list(img_shape)}")
print(f"  ✅ Channels     : 4 — FLAIR, T1, T1ce, T2")
print(f"  ✅ Label values : {unique_lbs}")
print(f"  ✅ Normalized   : per modality nonzero")
print(f"  ✅ Augmented    : train only (pos/neg crop + flip + rotate)")
print()
print("  Ready for Cell 9 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Build UNETR Model, Optimiser, Scheduler, Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Instantiate all four training components:
#
#   1. UNETR Model (UNet Transformers, WACV 2022):
#      A hybrid CNN + Vision Transformer segmentation architecture.
#      The ViT encoder splits the 128³ volume into 16³ patches and
#      extracts global context via self-attention across all patches.
#      Skip connections feed intermediate ViT representations into a
#      CNN decoder that recovers full spatial resolution step by step.
#      Configured for BraTS 2020:
#        in_channels=4  — all four MRI modalities stacked as channels
#        out_channels=4 — background, NCR, edema, enhancing tumour
#        img_size=128³  — matches the crop size from Cell 7
#
#   2. Optimiser — Adam (lr=1e-4, weight_decay=1e-5):
#      Adaptive learning rate per parameter. weight_decay adds L2
#      regularisation to reduce overfitting on the small BraTS dataset.
#
#   3. Scheduler — CosineAnnealingLR:
#      FIX-B: T_max=MAX_EPOCHS (25).
#      Original T_max=100 only completed 25% of the cosine decay curve
#      across 25 epochs, providing almost no LR reduction benefit.
#      T_max=25 gives one full cosine cycle: lr smoothly decays from
#      1e-4 down to eta_min=1e-6 by the final epoch.
#
#   4. Loss — DiceCELoss (equal Dice + CrossEntropy weighting):
#      Dice term handles severe class imbalance (tumour << background).
#      CE term stabilises gradient flow early in training when Dice
#      gradients are weak because predictions are near-uniform.
#      include_background=False excludes the dominant background class
#      so the loss focuses on the three tumour classes only.
#
#   
# ─────────────────────────────────────────────────────────────────────────────

model = UNETR(
    in_channels   = 4,                   # FLAIR, T1, T1ce, T2
    out_channels  = 4,                   # BG, NCR, Edema, ET
    img_size      = (128, 128, 128),
    feature_size  = 16,
    hidden_size   = 768,
    mlp_dim       = 3072,
    num_heads     = 12,
    proj_type     = 'perceptron',
    norm_name     = 'batch',             # FIX-A: was 'instance'
    conv_block    = True,
    res_block     = True,
    dropout_rate  = 0.2,
).to(device)

optimizer = Adam(
    model.parameters(),
    lr=1e-4, weight_decay=1e-5)

# FIX-B: T_max=MAX_EPOCHS — full cosine cycle within the 25-epoch budget.
scheduler = CosineAnnealingLR(
    optimizer,
    T_max   = MAX_EPOCHS,
    eta_min = 1e-6)

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
print("CELL 9 — MODEL READY — UNETR")
print("=" * 55)
print(f"  Architecture     : UNETR (WACV 2022)")
print(f"  Dataset          : {DATASET_TYPE}")
print()
print("  Input / Output:")
print(f"    In  channels   : 4  (FLAIR, T1, T1ce, T2)")
print(f"    Out channels   : 4  (BG, NCR, Edema, ET)")
print(f"    Image size     : 128×128×128")
print()
print("  ViT Encoder:")
print(f"    Patch size     : 16×16×16")
print(f"    ViT layers     : 12")
print(f"    Hidden size    : 768")
print(f"    MLP dim        : 3072")
print(f"    Attn heads     : 12")
print(f"    Feature size   : 16")
print(f"    Dropout        : 0.2")
print(f"    Norm           : BatchNorm  ← FIX-A (was instance)")
print()
print("  Training:")
print(f"    Optimizer      : Adam  lr=1e-4  wd=1e-5")
print(f"    Scheduler      : CosineAnnealingLR")
print(f"    T_max          : {MAX_EPOCHS}  ← FIX-B (full cosine cycle)")
print(f"    eta_min        : 1e-6")
print(f"    Loss           : DiceCELoss  λ_dice=0.5  λ_ce=0.5")
print(f"    Bg excluded    : True (loss focuses on tumour classes)")
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
#   Metrics are created once here and reused every validation epoch —
#   they accumulate results across all batches then aggregate at the end.
#
#   Primary metrics (per tumour class, background excluded):
#     DiceMetric            — measures voxel overlap between prediction and
#                             ground truth. Main BraTS ranking metric.
#                             Range 0 (no overlap) to 1 (perfect overlap).
#     MeanIoU               — Jaccard index, similar to Dice but more
#                             sensitive to small region errors.
#     ConfusionMatrixMetric — derives sensitivity (recall), specificity,
#                             and precision from the confusion matrix.
#
#   BraTS clinical sub-region metrics:
#     These three are the official BraTS leaderboard metrics and must be
#     reported in any BraTS comparison paper.
#     Whole Tumour  (WT) = NCR + Edema + ET   channels 1+2+3
#     Tumour Core   (TC) = NCR + ET            channels 1+3
#     Enhancing Tumour (ET) = ET only          channel 3
#
#   LOGIC-2 FIX — include_background=True for WT / TC / ET metrics:
#     Sub-region tensors passed to these metrics are single-channel
#     binary maps of shape (1, 1, D, H, W).
#     With include_background=False MONAI treats that single channel
#     as background and skips it entirely, returning NaN silently.
#     include_background=True tells MONAI to score channel 0, which
#     is the binary tumour region we actually want to measure.
#
#   HD95 is intentionally NOT instantiated here:
#     Hausdorff Distance 95th percentile requires computing nearest-
#     neighbour surface distances on full 3-D volumes for every class
#     and every sample. This is 3-5x slower than Dice computation.
#     Running it every validation epoch would make 25-epoch training
#     impractical on a T4 GPU. It is reserved for the final eval cell.
#     Its column in all CSVs is filled with float('nan') — LOGIC-3 FIX.
#
#   Post-processing transforms:
#     post_pred  — argmax across class dim → one-hot encode (4 classes)
#     post_label — integer label map → one-hot encode (4 classes)
#     Both are needed because DiceMetric expects one-hot inputs.
# ─────────────────────────────────────────────────────────────────────────────

dice_metric = DiceMetric(
    include_background = False,      # exclude BG, score 3 tumour classes
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

# LOGIC-2 FIX: include_background=True because sub-region tensors are
# single-channel (1,1,D,H,W). include_background=False would skip the
# only channel, treating it as background, and return NaN every epoch.
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
print()
print("  Primary metrics  (include_background=False):")
print("    DiceMetric           ✓  main segmentation metric")
print("    MeanIoU              ✓  Jaccard index")
print("    ConfusionMatrix      ✓  sensitivity, specificity, precision")
print()
print("  BraTS sub-region metrics  (include_background=True ← LOGIC-2 FIX):")
print("    Dice WT  ✓  ch1+ch2+ch3  NCR+Edema+ET  (whole tumour)")
print("    Dice TC  ✓  ch1+ch3      NCR+ET         (tumour core)")
print("    Dice ET  ✓  ch3          ET only        (enhancing tumour)")
print()
print("  HD95:")
print("    Skipped during training  — too slow per epoch")
print("    val_hd95 = float('nan') in all CSVs  ← LOGIC-3 FIX")
print("    Computed in final evaluation cell only")
print()
print("  Post-processing:")
print("    post_pred   ✓  argmax → one-hot(4)")
print("    post_label  ✓  integer label → one-hot(4)")
print()
print("  All metrics defined ✓")
print("  Ready for Cell 11 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — Measure Efficiency Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Measure and record four model efficiency statistics that will appear
#   in the paper's comparison table alongside Dice and HD95:
#
#     params_M       — total parameter count in millions
#                      computed from model.parameters(), not estimated.
#     model_size_MB  — on-disk memory footprint of weights in float32.
#                      formula: total_params × 4 bytes / 1024² = MB.
#     inference_ms   — average wall-clock time for one forward pass on
#                      a single 128×128×128 volume in milliseconds.
#     gpu_memory_MB  — peak GPU memory allocated during one inference,
#                      measured via torch.cuda.max_memory_allocated().
#
#   Measurement methodology:
#     3 warmup runs  — allow the GPU to reach steady-state clock speed,
#                      warm up CUDA kernel caches, and trigger any JIT
#                      compilation so timing is not contaminated by
#                      one-time startup costs.
#     10 timed runs  — mean taken across 10 runs to reduce variance
#                      from GPU scheduler jitter and memory bandwidth
#                      fluctuations.
#     CUDA Events    — used instead of Python time.time() because they
#                      measure GPU-side execution time directly, which
#                      is more accurate for async GPU operations.
#
#   MINOR-1 FIX — GPU memory freed after measurement:
#     Original code did not delete the dummy tensor or call empty_cache()
#     after the efficiency loop. The dummy tensor stayed allocated on GPU,
#     consuming memory before training even started. Fixed by explicitly
#     deleting the tensor and calling torch.cuda.empty_cache() so the
#     full GPU memory budget is available for training.
# ─────────────────────────────────────────────────────────────────────────────

def measure_efficiency(model, device,
                       input_shape = (1, 4, 128, 128, 128),
                       n_warmup    = 3,
                       n_runs      = 10):
    model.eval()
    dummy        = torch.zeros(*input_shape).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    if device.type == 'cuda':
        # Reset peak stats so previous allocations don't inflate reading
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        # Warmup runs — not timed
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
        torch.cuda.synchronize()

        # Reset again after warmup so peak reflects inference only
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
        # CPU fallback — use Python timer
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

    # MINOR-1 FIX: delete dummy tensor and clear GPU cache so full memory
    # budget is available when training starts in Cell 15.
    del dummy
    torch.cuda.empty_cache()

    return {
        'total_params'  : total_params,
        'params_M'      : round(total_params / 1e6,            4),
        'model_size_MB' : round(total_params * 4 / 1024 / 1024, 2),
        'inference_ms'  : round(inference_ms,                   2),
        'gpu_memory_MB' : round(gpu_memory_MB,                  2),
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
print("  GPU cache cleared after measurement ✓  (MINOR-1 fix)")
print("  Efficiency values ready for paper table ✓")
print("  Ready for Cell 12 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — Model Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Print a complete layer-by-layer summary of the UNETR model using
#   torchinfo. This serves three purposes:
#
#   1. Shape verification — confirms that input tensor (1, 4, 128, 128, 128)
#      flows correctly through every layer and the final output shape is
#      (1, 4, 128, 128, 128) matching out_channels=4 and the crop size.
#
#   2. Parameter audit — shows parameter count per layer so any unexpected
#      layer (wrong size, extra layer, missing layer) is immediately visible.
#      Total should match eff['total_params'] computed in Cell 11.
#
#   3. Trainable flag — the 'trainable' column confirms all layers are
#      set to train. If any layer were accidentally frozen (e.g. from a
#      partial checkpoint load), it would show False here.
#
#   The dummy input is created fresh, passed to summary(), then immediately
#   deleted and GPU cache cleared so no extra memory is held going into
#   the training loop in Cell 15.
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

# Free GPU memory immediately — do not carry dummy tensor into training
del dummy_input
torch.cuda.empty_cache()

print()
print("=" * 55)
print("SUMMARY VERIFICATION")
print("=" * 55)
print(f"  Expected input  : (1, 4, 128, 128, 128)")
print(f"  Expected output : (1, 4, 128, 128, 128)")
print(f"  Expected params : {eff['params_M']:.4f} M")
print(f"  Expected device : {device}")
print()
print("  Model summary printed   ✓")
print("  GPU cache cleared       ✓")
print("  Ready for Cell 13       ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — Training & Validation Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define all functions called inside the training loop in Cell 15.
#   Keeping them here (not inside the loop) means they are defined once,
#   are easy to unit-test individually, and keep Cell 15 readable.
#
#   Four functions are defined:
#
#   1. extract_subregions(pred_onehot, label_onehot):
#        Derives the three BraTS clinical sub-region binary maps from
#        four-class one-hot tensors of shape (4, D, H, W).
#        WT = ch1 + ch2 + ch3   whole tumour
#        TC = ch1 + ch3         tumour core
#        ET = ch3               enhancing tumour
#        Each result is wrapped to (1, 1, D, H, W) for DiceMetric.
#        BUG-2 FIX: ndim assertion added so a wrong-shaped tensor raises
#        a clear error immediately rather than causing a cryptic index
#        failure deep inside the validation loop.
#
#   2. extract_conf(result, idx):
#        Safely extracts a scalar value from ConfusionMatrixMetric
#        .aggregate() output. MONAI returns different formats across
#        versions (tuple, list of tensors, or a plain 2-D tensor).
#        This helper handles all three formats so the validation function
#        works regardless of which MONAI version is installed.
#
#   3. train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
#        Runs one full pass over the training DataLoader:
#          • forward pass through UNETR
#          • DiceCELoss computation
#          • gradient clipping at max_norm=1.0 to prevent exploding
#            gradients in the ViT attention layers
#          • Adam parameter update
#        Returns mean epoch loss and wall-clock time in minutes.
#        Prints batch loss every 50 steps to confirm progress.
#
#   4. validate(model, loader, ...):
#        Runs sliding window inference on full volumes (no cropping)
#        and computes all 8 metrics across the entire val set.
#        FIX-D   : overlap=0.5 (was 0.75) — standardised to BraTS 2023.
#        LOGIC-3 : val_hd95=float('nan') — clean missing-value marker.
#                  HD95 is computed only in the final evaluation cell.
#        All six metric accumulators are reset after aggregation so
#        they start clean for the next validation epoch.
#
#   A sanity test at the end verifies that extract_conf() handles the
#   ConfusionMatrixMetric return format of the installed MONAI version.
# ─────────────────────────────────────────────────────────────────────────────

def extract_subregions(pred_onehot, label_onehot):
    """
    Input  : (4, D, H, W) one-hot tensors — no batch dimension.
    Output : 6 tensors of shape (1, 1, D, H, W)
               pred_wt,  label_wt   whole tumour
               pred_tc,  label_tc   tumour core
               pred_et,  label_et   enhancing tumour
    BUG-2 FIX: ndim checked so shape errors surface with a clear message.
    """
    # BUG-2 FIX: guard both ndim and channel count
    assert pred_onehot.ndim == 4, \
        f"Expected 4-D tensor (C,D,H,W), got ndim={pred_onehot.ndim}"
    assert pred_onehot.shape[0] == 4, \
        f"Expected 4 one-hot channels, got {pred_onehot.shape[0]}"

    # Whole Tumour — NCR + Edema + ET
    pred_wt  = (pred_onehot[1] + pred_onehot[2]
                + pred_onehot[3]).clamp(0, 1)
    label_wt = (label_onehot[1] + label_onehot[2]
                + label_onehot[3]).clamp(0, 1)

    # Tumour Core — NCR + ET
    pred_tc  = (pred_onehot[1] + pred_onehot[3]).clamp(0, 1)
    label_tc = (label_onehot[1] + label_onehot[3]).clamp(0, 1)

    # Enhancing Tumour — ET only
    pred_et  = pred_onehot[3]
    label_et = label_onehot[3]

    def wrap(t):
        # (D,H,W) → (1, 1, D, H, W)  — batch dim + channel dim for DiceMetric
        return t.unsqueeze(0).unsqueeze(0)

    return (wrap(pred_wt),  wrap(label_wt),
            wrap(pred_tc),  wrap(label_tc),
            wrap(pred_et),  wrap(label_et))


def extract_conf(result, idx):
    """
    Safely extract a float scalar from ConfusionMatrixMetric.aggregate()
    regardless of MONAI version return format:
      Format A — tuple of (tensor, not_nans)   [get_not_nans=True]
      Format B — list of plain tensors
      Format C — single 2-D tensor  (older MONAI)
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
    One full training epoch over the DataLoader.
    Returns (mean_loss, epoch_time_minutes).
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

        # Gradient clipping — prevents exploding gradients in ViT
        # attention layers, which can destabilise early training.
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
    Computes 8 metrics across the entire validation set.
    FIX-D   : overlap=0.5  (was 0.75)
    LOGIC-3 : val_hd95=float('nan') — computed in final eval cell only.
    """
    model.eval()
    val_loss = 0.0
    steps    = 0

    with torch.no_grad():
        for batch in loader:
            steps  += 1
            images  = batch['image'].to(device)
            labels  = batch['label'].to(device)

            # Sliding window inference — runs the model on overlapping
            # 128³ crops of the full volume and stitches predictions
            # together using Gaussian weighting at overlap regions.
            outputs = sliding_window_inference(
                images,
                roi_size      = (128, 128, 128),
                sw_batch_size = 4,
                predictor     = model,
                overlap       = 0.5,       # FIX-D: was 0.75
                mode          = 'gaussian',
            )

            val_loss += loss_fn(outputs, labels).item()

            # Post-process: logits → one-hot predictions and labels
            outputs_post = [post_pred(o)  for o in outputs]
            labels_post  = [post_label(l) for l in labels]

            # Accumulate primary metrics
            dice_metric(y_pred=outputs_post, y=labels_post)
            iou_metric( y_pred=outputs_post, y=labels_post)
            confusion_metric(y_pred=outputs_post, y=labels_post)

            # Accumulate sub-region metrics
            for pred_oh, label_oh in zip(outputs_post, labels_post):
                (p_wt, l_wt,
                 p_tc, l_tc,
                 p_et, l_et) = extract_subregions(pred_oh, label_oh)
                dice_wt(y_pred=p_wt, y=l_wt)
                dice_tc(y_pred=p_tc, y=l_tc)
                dice_et(y_pred=p_et, y=l_et)

    # Aggregate accumulated results across all batches
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

    # Reset all accumulators for next validation epoch
    for m in [dice_metric, iou_metric, confusion_metric,
              dice_wt, dice_tc, dice_et]:
        m.reset()

    return {
        'val_loss' : val_loss / steps,
        'val_dice' : val_dice,
        'val_iou'  : val_iou,
        'val_hd95' : float('nan'),    # LOGIC-3 FIX: NaN not -1.0
        'val_sens' : val_sens,
        'val_spec' : val_spec,
        'val_prec' : val_prec,
        'val_wt'   : val_wt,
        'val_tc'   : val_tc,
        'val_et'   : val_et,
    }


# ── Sanity test — ConfusionMatrixMetric format ────────────────────────────────
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
print("    extract_subregions()  ✓  ndim + channel guard  (BUG-2 fix)")
print("    extract_conf()        ✓  multi-format safe")
print("    train_one_epoch()     ✓  grad clip max_norm=1.0")
print("    validate()            ✓  overlap=0.5  (FIX-D)")
print("                             val_hd95=float('nan')  (LOGIC-3 fix)")
print("                             all metrics reset after aggregation")
print()
print("  All training functions ready ✓")
print("  Ready for Cell 14 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14 — Define save_logs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Define the save_logs() function that writes four CSV files to Google
#   Drive after every validation epoch. Writing after each validation
#   (not just at the end) means progress is preserved even if the Colab
#   session disconnects or times out mid-training.
#
#   Four CSV files are written on every call:
#
#   train_log.csv:
#     One row per training epoch. Contains epoch number, mean train loss,
#     current learning rate, and wall-clock time for that epoch in minutes.
#     Written every epoch so LR decay and loss trend are always up to date.
#
#   val_log.csv:
#     One row per validation epoch (every VAL_INTERVAL epochs).
#     Contains all 10 validation metrics: loss, Dice, IoU, HD95 (NaN),
#     sensitivity, specificity, precision, Dice WT, Dice TC, Dice ET.
#     FIX-7: uses history['val_epoch'] list for epoch numbers so val rows
#     are never misaligned with train rows due to index arithmetic errors.
#
#   full_log.csv:
#     One row per training epoch merging train and val columns.
#     Train-only epochs have None in all val columns.
#     Val epochs have both train and val columns filled.
#     Alignment is done via a dict lookup on val_epoch — not by index —
#     so it is correct regardless of VAL_INTERVAL value.  (FIX-7)
#
#   efficiency_log.csv:
#     One row per training epoch with static efficiency columns
#     (params_M, model_size_MB, inference_ms, gpu_memory_MB) repeated
#     alongside the per-epoch train_time_min for easy paper table export.
#     A final MEAN summary row is appended for the paper table.
#     FIX-E  : 'model' and 'dataset' columns added so multiple models'
#              efficiency logs can be concatenated into one CSV for
#              cross-model comparison without losing which row is which.
#     BUG-3 FIX: mean_time is computed BEFORE the list comprehension.
#              Python leaks the final values of loop variables i and ep
#              into the outer scope after a list comprehension. The original
#              code used those leaked variables in the summary dict, writing
#              the last epoch's train_time_min instead of the actual mean.
#              Computing mean_time before the comprehension and using it
#              explicitly in the summary dict fixes this completely.
# ─────────────────────────────────────────────────────────────────────────────

def save_logs(history, log_train, log_val,
              log_full, log_eff, eff_static):

    # ── train_log.csv ─────────────────────────────────────────────────────────
    pd.DataFrame({
        'epoch'          : history['epoch'],
        'train_loss'     : history['train_loss'],
        'lr'             : history['lr'],
        'train_time_min' : history['train_time_min'],
    }).to_csv(log_train, index=False)

    # ── val_log.csv ───────────────────────────────────────────────────────────
    if len(history['val_loss']) > 0:
        pd.DataFrame({
            'epoch'    : history['val_epoch'],    # FIX-7: explicit val epochs
            'val_loss' : history['val_loss'],
            'val_dice' : history['val_dice'],
            'val_iou'  : history['val_iou'],
            'val_hd95' : history['val_hd95'],     # float('nan') — LOGIC-3 fix
            'val_sens' : history['val_sens'],
            'val_spec' : history['val_spec'],
            'val_prec' : history['val_prec'],
            'val_wt'   : history['val_wt'],
            'val_tc'   : history['val_tc'],
            'val_et'   : history['val_et'],
        }).to_csv(log_val, index=False)

    # ── full_log.csv ──────────────────────────────────────────────────────────
    # Build a lookup dict {val_epoch_number: index_in_val_lists}
    # so we can fill val columns for the correct train row. (FIX-7)
    val_lookup = {ve: i for i, ve in enumerate(history['val_epoch'])}

    full_rows = []
    for i, ep in enumerate(history['epoch']):
        row = {
            'epoch'          : ep,
            'train_loss'     : history['train_loss'][i],
            'lr'             : history['lr'][i],
            'train_time_min' : history['train_time_min'][i],
            # Val columns default to None for train-only epochs
            'val_loss' : None, 'val_dice' : None,
            'val_iou'  : None, 'val_hd95' : None,
            'val_sens' : None, 'val_spec' : None,
            'val_prec' : None, 'val_wt'   : None,
            'val_tc'   : None, 'val_et'   : None,
        }
        # If this epoch had validation, fill in all val columns
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

    # ── efficiency_log.csv ────────────────────────────────────────────────────
    # BUG-3 FIX: compute mean_time here, BEFORE the list comprehension below.
    # The comprehension `for i, ep in enumerate(...)` leaks i and ep into
    # the outer scope in Python. Using them after the comprehension would write
    # the last epoch's value instead of the mean into the summary row.
    mean_time = round(np.mean(history['train_time_min']), 4)

    eff_rows = [{
        'model'          : MODEL_NAME,        # FIX-E: model identity column
        'dataset'        : DATASET_TYPE,      # FIX-E: dataset identity column
        'params_M'       : eff_static['params_M'],
        'model_size_MB'  : eff_static['model_size_MB'],
        'inference_ms'   : eff_static['inference_ms'],
        'gpu_memory_MB'  : eff_static['gpu_memory_MB'],
        'epoch'          : ep,
        'train_time_min' : history['train_time_min'][i],
    } for i, ep in enumerate(history['epoch'])]

    # Summary row — mean train time across all epochs for paper table
    summary_row = pd.DataFrame([{
        'model'          : MODEL_NAME,
        'dataset'        : DATASET_TYPE,
        'params_M'       : eff_static['params_M'],
        'model_size_MB'  : eff_static['model_size_MB'],
        'inference_ms'   : eff_static['inference_ms'],
        'gpu_memory_MB'  : eff_static['gpu_memory_MB'],
        'epoch'          : 'MEAN',
        'train_time_min' : mean_time,          # BUG-3 FIX: true mean
    }])

    pd.concat(
        [pd.DataFrame(eff_rows), summary_row],
        ignore_index=True
    ).to_csv(log_eff, index=False)


print("=" * 55)
print("CELL 14 — SAVE_LOGS DEFINED")
print("=" * 55)
print(f"  Model column    : {MODEL_NAME}    (FIX-E)")
print(f"  Dataset column  : {DATASET_TYPE}  (FIX-E)")
print()
print("  4 CSV files written per validation epoch:")
print(f"    train_log.csv  → {LOG_TRAIN}")
print(f"    val_log.csv    → {LOG_VAL}")
print(f"    full_log.csv   → {LOG_FULL}")
print(f"    eff_log.csv    → {LOG_EFF}")
print()
print("  Fixes applied:")
print("    val_epoch list for row alignment   ✓  (FIX-7)")
print("    model + dataset columns            ✓  (FIX-E)")
print("    mean_time before comprehension     ✓  (BUG-3 fix)")
print("    val_hd95 = float('nan') in logs   ✓  (LOGIC-3 fix)")
print()
print("  save_logs defined ✓")
print("  Ready for Cell 15 ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15 — Training Loop (25 Epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE:
#   Run the main training loop for MAX_EPOCHS (25) epochs.
#   This cell orchestrates all components defined in Cells 9-14:
#     • train_one_epoch()  — forward + backward + optimiser step
#     • scheduler.step()  — cosine LR decay after each epoch
#     • validate()        — full sliding window inference on val set
#     • save_logs()       — write all four CSVs to Google Drive
#     • torch.save()      — persist best model and periodic checkpoints
#
#   Per-epoch flow:
#     1. Train one epoch → mean loss + wall-clock time
#     2. Step LR scheduler
#     3. Append all train metrics to history dict
#     4. Every VAL_INTERVAL (2) epochs:
#          a. Run full validation → 10 metrics
#          b. Append val metrics to history using val_epoch tracker (FIX-7)
#          c. Write all four CSVs to Drive (progress preserved on disconnect)
#          d. If val Dice improved → overwrite best_model.pth
#     5. Every CKPT_INTERVAL (2) epochs:
#          • Save numbered checkpoint epoch_NNN.pth for manual recovery
#
#   History dict design:
#     Two separate epoch trackers are maintained:
#       history['epoch']     — every training epoch  [1,2,3,...,25]
#       history['val_epoch'] — only val epochs       [2,4,6,...,24]
#     This separation (FIX-7) allows save_logs() to correctly align
#     train and val rows in full_log.csv regardless of VAL_INTERVAL.
#
#   Checkpoint design:
#     Both best_model.pth and epoch_NNN.pth include 'model_name' and
#     'dataset' keys (FIX-F) so loading code can verify the checkpoint
#     matches the architecture before restoring weights.
#
#   Final save_logs() call after the loop ensures the last epoch's
#   data is written even if it was not a validation epoch.
# ─────────────────────────────────────────────────────────────────────────────

history = {
    'epoch'          : [],
    'train_loss'     : [],
    'lr'             : [],
    'train_time_min' : [],
    'val_epoch'      : [],    # FIX-7: explicit val-epoch tracker
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
print("  Fixes active:")
print(f"    Crop + unwrap_crop_list   ✓  (BUG-1 fix)")
print(f"    norm_name = batch         ✓  (FIX-A)")
print(f"    T_max = {MAX_EPOCHS}               ✓  (FIX-B)")
print(f"    overlap = 0.5             ✓  (FIX-D)")
print(f"    val_hd95 = float('nan')  ✓  (LOGIC-3 fix)")
print(f"    val_epoch tracker         ✓  (FIX-7)")
print(f"    model+dataset in ckpt     ✓  (FIX-F)")
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

    # ── 1. Train one epoch ────────────────────────────────────────────────────
    train_loss, epoch_time = train_one_epoch(
        model, train_loader,
        optimizer, loss_fn,
        device, epoch)

    # ── 2. Step LR scheduler ─────────────────────────────────────────────────
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # ── 3. Record train metrics ───────────────────────────────────────────────
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['lr'].append(lr)
    history['train_time_min'].append(round(epoch_time, 4))

    print(f"\n  Train Loss  : {train_loss:.4f}")
    print(f"  LR          : {lr:.7f}")
    print(f"  Epoch time  : {epoch_time:.2f} min")

    # ── 4. Validation ─────────────────────────────────────────────────────────
    if epoch % VAL_INTERVAL == 0:

        print(f"\n  Running validation...")

        metrics = validate(
            model, val_loader, loss_fn,
            dice_metric, iou_metric,
            confusion_metric,
            dice_wt, dice_tc, dice_et,
            post_pred, post_label, device)

        # FIX-7: append to val_epoch separately from epoch
        history['val_epoch'].append(epoch)
        for k in ['val_loss', 'val_dice', 'val_iou', 'val_hd95',
                  'val_sens', 'val_spec', 'val_prec',
                  'val_wt',   'val_tc',   'val_et']:
            history[k].append(metrics[k])

        print(f"\n  ── Segmentation Metrics ─────────────────")
        print(f"  Val Loss    : {metrics['val_loss']:.4f}")
        print(f"  Dice        : {metrics['val_dice']:.4f}"
              f"  (Best so far: {max(history['val_dice']):.4f})")
        print(f"  IoU         : {metrics['val_iou']:.4f}")
        print(f"  HD95        : eval cell only"
              f"  (NaN in logs — LOGIC-3 fix)")
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

        # Write all four CSVs — progress saved even on disconnect
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
                'model_name' : MODEL_NAME,    # FIX-F
                'dataset'    : DATASET_TYPE,  # FIX-F
            }, CKPT_DIR + 'best_model.pth')
            print(f"  ✓ Best model saved!")
            print(f"    Dice  = {best_dice:.4f}")
            print(f"    Epoch = {best_epoch}")

    # ── 5. Periodic checkpoint ────────────────────────────────────────────────
    if epoch % CKPT_INTERVAL == 0:
        ckpt_path = CKPT_DIR + f'epoch_{epoch:03d}.pth'
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_dice'  : best_dice,
            'efficiency' : eff,
            'history'    : history,
            'model_name' : MODEL_NAME,    # FIX-F
            'dataset'    : DATASET_TYPE,  # FIX-F
        }, ckpt_path)
        print(f"  Checkpoint saved → epoch_{epoch:03d}.pth ✓")

# ── Final CSV save ────────────────────────────────────────────────────────────
# Ensures last epoch data is written even if it was not a val epoch
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

#### If any disconnect, then first see the check points status then resume cell and then training cell

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINT STATUS — Run first to verify
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 55)
print("CHECKPOINT STATUS — UNETR")
print("=" * 55)

checkpoints = sorted(glob.glob(CKPT_DIR + 'epoch_*.pth'))
print(f"  Epoch checkpoints : {len(checkpoints)}")
for ck in checkpoints:
    mb   = os.path.getsize(ck) / 1024 / 1024
    data = torch.load(ck, map_location='cpu', weights_only=False)
    patch_key = 'vit.patch_embedding.patch_embeddings.1.weight'
    shape     = data['model_state'][patch_key].shape
    in_ch     = shape[1] // 4096
    print(f"    {os.path.basename(ck)}"
          f"  {mb:.1f} MB"
          f"  in_ch={in_ch}"
          f"  epoch={data['epoch']}"
          f"  dice={data['best_dice']:.4f}"
          f"  {'✅' if in_ch == 4 else '❌ wrong'}")

print()
best_path = CKPT_DIR + 'best_model.pth'
if os.path.exists(best_path):
    ck = torch.load(best_path, map_location='cpu',
                    weights_only=False)
    print(f"  best_model.pth  ✓")
    print(f"    Epoch     : {ck['epoch']}")
    print(f"    Best Dice : {ck['best_dice']:.4f}")
else:
    print("  best_model.pth  ✗")

print()
for log in [LOG_TRAIN, LOG_VAL, LOG_FULL, LOG_EFF]:
    if os.path.exists(log):
        df = pd.read_csv(log)
        print(f"  {os.path.basename(log)}"
              f"  —  {len(df)} rows ✓")
    else:
        print(f"  {os.path.basename(log)}  —  not found")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESUME CELL — UNETR BraTS2020
# Auto-finds latest 4-channel checkpoint
# Safe to run after every disconnect
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import gc

# Load eff
best_path = CKPT_DIR + 'best_model.pth'
if os.path.exists(best_path):
    best_ck = torch.load(
        best_path, map_location='cpu', weights_only=False)
    eff = best_ck.get('efficiency', {
        'params_M'      : 92.8,
        'model_size_MB' : 371.2,
        'inference_ms'  : 0.0,
        'gpu_memory_MB' : 0.0,
    })
    print("  eff loaded from best_model.pth ✓")
else:
    eff = {
        'params_M'      : 92.8,
        'model_size_MB' : 371.2,
        'inference_ms'  : 0.0,
        'gpu_memory_MB' : 0.0,
    }
    print("  using default eff ✓")

# Find latest 4-channel compatible checkpoint automatically
all_ckpts = sorted(glob.glob(CKPT_DIR + 'epoch_*.pth'))

# Filter — only keep 4-channel checkpoints
valid_ckpts = []
patch_key   = 'vit.patch_embedding.patch_embeddings.1.weight'
for ck_path in all_ckpts:
    ck    = torch.load(ck_path, map_location='cpu',
                       weights_only=False)
    shape = ck['model_state'][patch_key].shape
    in_ch = shape[1] // 4096
    if in_ch == 4:
        valid_ckpts.append(ck_path)
        print(f"  ✅ {os.path.basename(ck_path)}"
              f"  in_ch={in_ch}"
              f"  epoch={ck['epoch']}"
              f"  dice={ck['best_dice']:.4f}")
    else:
        print(f"  ❌ {os.path.basename(ck_path)}"
              f"  in_ch={in_ch}  skipped")

if len(valid_ckpts) == 0:
    raise ValueError(
        "No 4-channel checkpoints found. "
        "Check CKPT_DIR or delete wrong checkpoints."
    )

# Use latest valid checkpoint
latest_ckpt = valid_ckpts[-1]
checkpoint  = torch.load(
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

# Restore scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max      = MAX_EPOCHS,
    eta_min    = 1e-6,
    last_epoch = checkpoint['epoch']
)

gc.collect()
torch.cuda.empty_cache()

print()
print("=" * 55)
print("RESUMED — UNETR BraTS2020")
print("=" * 55)
print(f"  Checkpoint   : {os.path.basename(latest_ckpt)}")
print(f"  Resumed from : epoch {checkpoint['epoch']}")
print(f"  Starting at  : epoch {START_EPOCH}")
print(f"  Remaining    : {MAX_EPOCHS - checkpoint['epoch']} epochs")
print(f"  Best Dice    : {best_dice:.4f}")
print(f"  History rows : {len(history['epoch'])} epochs")
print(f"  LR restored  : {optimizer.param_groups[0]['lr']:.7f}")
print()
print("  model weights     ✓")
print("  optimiser state   ✓")
print("  history dict      ✓")
print("  scheduler         ✓")
print("  in_channels=4     ✓  verified")
print("  GPU cache cleared ✓")
print()
print("  Now run the training loop cell ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP — Resumed from START_EPOCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
              f"  (Best so far: {max(history['val_dice']):.4f})")
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

        save_logs(history,
                  LOG_TRAIN, LOG_VAL,
                  LOG_FULL,  LOG_EFF, eff)
        print(f"\n  CSVs saved to Drive ✓")

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
            print(f"  ✓ Best model saved!")
            print(f"    Dice  = {best_dice:.4f}")
            print(f"    Epoch = {best_epoch}")

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
        print(f"  Checkpoint saved → epoch_{epoch:03d}.pth ✓")

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