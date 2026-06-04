# ============================================================
# preprocessing.py
# Shared dataset preprocessing for Alzheimer's MRI project
# UCI CS 274P | Spring 2026
#
# ── COLAB USAGE ──────────────────────────────────────────────
#   Step 1: Upload kaggle.json when prompted
#   Step 2: In your notebook:
#
#     !wget -q https://raw.githubusercontent.com/wj6801/COMPSCI-274P/main/preprocessing.py
#     from preprocessing import setup_dataset
#     train_loader, val_loader, test_loader, meta = setup_dataset()
#
# ── LOCAL USAGE ──────────────────────────────────────────────
#   Step 1: Place kaggle.json in ~/.kaggle/kaggle.json
#   Step 2: pip install kaggle torch torchvision scikit-learn
#   Step 3: python preprocessing.py   (to test)
#   Step 4: from preprocessing import setup_dataset
#
# ── WHAT THIS SCRIPT DOES ────────────────────────────────────
#   1. Downloads OASIS dataset via Kaggle API
#   2. Filters to axial-only slices (middle z-range, depth 60-120)
#   3. Creates subject-level train/val/test split (seed=42)
#   4. Saves split_indices.json (shared across all team members)
#   5. Returns DataLoaders with WeightedRandomSampler
#
# ── WHY SUBJECT-LEVEL SPLIT ──────────────────────────────────
#   Slice-level splitting causes data leakage: the same patient's
#   slices appear in both train and test, giving ~99.97% accuracy
#   that reflects patient memorization, not disease generalization.
#   Subject-level splitting ensures each patient appears in exactly
#   one partition, giving honest generalization estimates.
#
# ── WHY AXIAL FILTER ─────────────────────────────────────────
#   OASIS provides slices at all z-levels. Upper slices (>120)
#   show only cortex with no subcortical structures. Lower slices
#   (<60) show brainstem/cerebellum irrelevant to Alzheimer's.
#   Middle-range axial slices (60-120) consistently show the
#   ventricles and periventricular structures most affected by
#   Alzheimer's progression.
# ============================================================

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError(
        "scikit-learn not found. Run: pip install scikit-learn"
    )


# ════════════════════════════════════════════════════════════
# CONFIGURATION
# Change data_dir if your local path is different.
# Everything else should stay the same across all team members.
# ════════════════════════════════════════════════════════════

def _is_colab():
    return os.path.exists("/content")

CONFIG = {
    # Dataset location
    "data_dir": "/content/data/oasis/" if _is_colab() else "./data/oasis/",

    # Split file — commit this to GitHub so everyone shares the same split
    "split_path": "split_indices.json",

    # Kaggle dataset
    "kaggle_dataset": "ninadaithal/imagesoasis",

    # Axial slice filter range (slice index in filename)
    # OAS1_0001_MR1_mpr-1_100.jpg → slice index = 100
    # Keep 60-120: ventricular + cortical level axial slices
    "slice_min": 60,
    "slice_max": 120,

    # Train/val/test ratios
    "train_ratio": 0.70,
    "random_seed": 42,

    # DataLoader settings
    "batch_size":   32,
    "image_size":   224,
    "num_workers":  2 if _is_colab() else 4,
}

# ── Class definitions (ImageFolder alphabetical order) ───────
# Folder names in the dataset:
CLASS_NAMES_FOLDER = [
    'Mild Dementia',
    'Moderate Dementia',
    'Non Demented',
    'Very mild Dementia',
]
# Cleaned names (same order — ImageFolder assigns 0,1,2,3):
CLASS_NAMES = [
    'MildDemented',       # index 0
    'ModerateDemented',   # index 1
    'NonDemented',        # index 2
    'VeryMildDemented',   # index 3
]
N_CLASSES = 4

# ImageNet normalization (used even for MRI — standard practice
# with pretrained models to preserve learned feature statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════
# STEP 1 — Download dataset
# ════════════════════════════════════════════════════════════

def download_dataset(data_dir=None):
    """
    Downloads the OASIS dataset via Kaggle API.

    Colab:  prompts you to upload kaggle.json
    Local:  expects kaggle.json at ~/.kaggle/kaggle.json
            Download from: kaggle.com → Account → API → Create Token
    """
    data_dir = data_dir or CONFIG["data_dir"]

    if os.path.exists(data_dir) and any(os.scandir(data_dir)):
        print(f"✓ Dataset already at {data_dir} — skipping download")
        return data_dir

    # ── Colab: prompt for kaggle.json upload ─────────────────
    if _is_colab():
        kaggle_cfg = "/root/.kaggle/kaggle.json"
        if not os.path.exists(kaggle_cfg):
            print("Upload your kaggle.json when prompted:")
            from google.colab import files
            files.upload()
            os.makedirs("/root/.kaggle", exist_ok=True)
            shutil.copy("kaggle.json", kaggle_cfg)
            os.chmod(kaggle_cfg, 0o600)
            print("✓ kaggle.json configured")
    # ── Local: check ~/.kaggle/kaggle.json exists ────────────
    else:
        kaggle_cfg = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_cfg):
            raise FileNotFoundError(
                f"\nkaggle.json not found at: {kaggle_cfg}\n"
                "To fix:\n"
                "  1. Go to kaggle.com → Account → API → Create New Token\n"
                "  2. Download kaggle.json\n"
                f"  3. Move it to {kaggle_cfg}\n"
                "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
            )

    # ── Download ──────────────────────────────────────────────
    tmp_dir = "/content/data" if _is_colab() else "./data"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Downloading {CONFIG['kaggle_dataset']} (~1-2 mins)...")
    exit_code = os.system(
        f"kaggle datasets download -d {CONFIG['kaggle_dataset']} "
        f"-p {tmp_dir} --quiet"
    )
    if exit_code != 0:
        raise RuntimeError(
            "Kaggle download failed. Check your kaggle.json and internet connection."
        )

    zip_path = os.path.join(tmp_dir, "imagesoasis.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Expected zip at {zip_path} but not found after download."
        )

    print("Unzipping...")
    os.makedirs(data_dir, exist_ok=True)
    os.system(f"unzip -q {zip_path} -d {data_dir}")
    os.remove(zip_path)
    print(f"✓ Dataset ready at {data_dir}")
    return data_dir


# ════════════════════════════════════════════════════════════
# STEP 2 — Locate the class folders
# ════════════════════════════════════════════════════════════

def find_data_dir(data_dir=None):
    """
    Walks the data directory to find the folder that directly
    contains the four class subfolders. Returns that path.
    """
    data_dir = data_dir or CONFIG["data_dir"]
    for root, dirs, _ in os.walk(data_dir):
        if any(c in dirs for c in CLASS_NAMES_FOLDER):
            print(f"✓ Class folders found at: {root}")
            return root
    raise FileNotFoundError(
        f"Could not find class folders under {data_dir}\n"
        f"Expected folders: {CLASS_NAMES_FOLDER}\n"
        f"Check that the download completed successfully."
    )


# ════════════════════════════════════════════════════════════
# STEP 3 — Axial slice filter
# ════════════════════════════════════════════════════════════

def is_axial_slice(filename):
    """
    Returns True if this file is an axial slice within the
    configured z-range (slice_min to slice_max).

    OASIS filename format:
        OAS1_0001_MR1_mpr-1_100.jpg
                              ^^^── slice index (100)

    The slice index is the last underscore-separated number
    before the file extension.

    Rationale for 60-120 range:
        < 60  : cerebellum, brainstem — not relevant for Alzheimer's
        60-120: ventricles, hippocampal level, periventricular cortex
                — primary structures affected by Alzheimer's
        > 120 : superior cortex only — misses subcortical structures
    """
    stem  = Path(filename).stem          # strip extension
    parts = stem.split('_')
    try:
        slice_idx = int(parts[-1])
        return CONFIG["slice_min"] <= slice_idx <= CONFIG["slice_max"]
    except (ValueError, IndexError):
        # Cannot parse slice number — keep the file to be safe
        return True


# ════════════════════════════════════════════════════════════
# STEP 4 — Subject-level split
# ════════════════════════════════════════════════════════════

def make_subject_split(full_dataset, split_path=None):
    """
    Creates (or loads) a subject-level train/val/test split.

    Subject ID is extracted from the filename:
        OAS1_0001_MR1_mpr-1_100.jpg → subject OAS1_0001

    All slices from one subject go entirely into one partition.
    This prevents data leakage where adjacent slices from the
    same patient appear in both train and test.

    Split: 70% train / 15% val / 15% test
    First split is stratified by class (enough subjects).
    Second split (val/test) is not stratified because
    Moderate Dementia has too few subjects (<2 in temp set).

    Returns:
        train_idx, val_idx, test_idx  (lists of dataset indices)
    """
    split_path = split_path or CONFIG["split_path"]

    if os.path.exists(split_path):
        print(f"✓ Loading split from {split_path}")
        with open(split_path) as f:
            split = json.load(f)
        ti, vi, tei = split['train'], split['val'], split['test']
        print(f"  Train: {len(ti)} | Val: {len(vi)} | Test: {len(tei)}")
        return ti, vi, tei

    print("Generating subject-level split (this happens once)...")

    # ── Group image indices by subject ID ─────────────────────
    subject_to_indices = defaultdict(list)
    for idx, (path, label) in enumerate(full_dataset.samples):
        fname      = os.path.basename(path)
        parts      = fname.split('_')
        subject_id = '_'.join(parts[:2]) if len(parts) >= 2 else fname
        subject_to_indices[subject_id].append((idx, label))

    subjects = list(subject_to_indices.keys())

    # ── Assign each subject its majority class label ───────────
    subject_labels = []
    for s in subjects:
        labels = [l for _, l in subject_to_indices[s]]
        subject_labels.append(max(set(labels), key=labels.count))

    print(f"\n  Total subjects: {len(subjects)}")
    print("  Subject-level class distribution:")
    for cls_idx, count in sorted(Counter(subject_labels).items()):
        print(f"    {CLASS_NAMES[cls_idx]:<25} {count} subjects")

    # ── First split: train vs temp ────────────────────────────
    train_subj, temp_subj = train_test_split(
        subjects,
        test_size=1 - CONFIG["train_ratio"],
        random_state=CONFIG["random_seed"],
        stratify=subject_labels,   # works here — enough subjects
    )

    # ── Second split: val vs test ─────────────────────────────
    # No stratify: Moderate Dementia has only ~2 subjects total,
    # leaving <2 in temp — sklearn requires ≥2 per class to stratify
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=0.5,
        random_state=CONFIG["random_seed"],
    )

    # ── Convert subject lists → image index lists ─────────────
    train_idx = [idx for s in train_subj
                 for idx, _ in subject_to_indices[s]]
    val_idx   = [idx for s in val_subj
                 for idx, _ in subject_to_indices[s]]
    test_idx  = [idx for s in test_subj
                 for idx, _ in subject_to_indices[s]]

    # ── Save ──────────────────────────────────────────────────
    split = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    with open(split_path, 'w') as f:
        json.dump(split, f)

    print(f"\n  Train: {len(train_idx)} images | {len(train_subj)} subjects")
    print(f"  Val:   {len(val_idx)} images | {len(val_subj)} subjects")
    print(f"  Test:  {len(test_idx)} images | {len(test_subj)} subjects")

    # ── Leakage check ─────────────────────────────────────────
    overlap = set(train_idx) & set(test_idx)
    assert len(overlap) == 0, \
        f"DATA LEAKAGE: {len(overlap)} images appear in both train and test!"
    print(f"✓ Leakage check passed — 0 images shared between train and test")
    print(f"✓ Split saved to {split_path} — commit this file to GitHub!")

    return train_idx, val_idx, test_idx


# ════════════════════════════════════════════════════════════
# STEP 5 — Transforms
# ════════════════════════════════════════════════════════════

def get_transforms(image_size=None):
    """
    Returns (train_transform, val_transform).

    Train: augmentation (flip, rotation, jitter, affine)
    Val/Test: resize + normalize only

    Uses ImageNet normalization even for MRI because pretrained
    models expect this normalization. The features transfer
    reasonably well despite the domain difference.
    """
    size = image_size or CONFIG["image_size"]

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


# ════════════════════════════════════════════════════════════
# STEP 6 — Custom Dataset with axial filter
# ════════════════════════════════════════════════════════════

class OASISSubset(torch.utils.data.Dataset):
    """
    Wraps an ImageFolder dataset with:
      - Index subsetting (train/val/test split)
      - Optional axial slice filtering by filename
      - Optional label remapping (for binary classification)

    Parameters
    ----------
    base_dataset : ImageFolder
        Full OASIS dataset loaded with ImageFolder.
    indices : list[int]
        Indices belonging to this partition (from split JSON).
    transform : torchvision.transforms
        Transform to apply to each image.
    filter_axial : bool
        If True, further filters indices to axial slice range.
    label_map : dict or None
        Optional remapping of class indices.
        Example for binary (Demented vs NonDemented):
            {0: 1, 1: 1, 2: 0, 3: 1}
    """
    def __init__(
        self,
        base_dataset,
        indices,
        transform=None,
        filter_axial=True,
        label_map=None,
    ):
        self.base      = base_dataset
        self.transform = transform
        self.label_map = label_map

        if filter_axial:
            original_count = len(indices)
            self.indices = [
                i for i in indices
                if is_axial_slice(
                    os.path.basename(base_dataset.samples[i][0])
                )
            ]
            print(f"    Axial filter: {original_count} → "
                  f"{len(self.indices)} images kept")
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        if self.label_map is not None:
            label = self.label_map[label]
        return img, label


# ════════════════════════════════════════════════════════════
# STEP 7 — Compute loss weights
# ════════════════════════════════════════════════════════════

def compute_loss_weights(train_dataset, n_classes, device):
    """
    Computes inverse-frequency class weights for CrossEntropyLoss.

    Formula: w_c = N / (C * n_c)
    where N = total training images, C = number of classes,
    n_c = images in class c.

    Higher weight → model penalized more for misclassifying
    that class → forces attention to rare classes like
    Moderate Dementia.
    """
    all_labels   = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_counts = Counter(all_labels)
    total        = sum(label_counts.values())

    weights = torch.tensor(
        [total / (n_classes * label_counts.get(i, 1))
         for i in range(n_classes)],
        dtype=torch.float
    ).to(device)

    return weights


# ════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — call this from your model notebook
# ════════════════════════════════════════════════════════════

def setup_dataset(
    data_dir=None,
    split_path=None,
    filter_axial=True,
    label_map=None,
    device=None,
    batch_size=None,
    verbose=True,
):
    """
    Full pipeline: download → filter → split → DataLoaders.

    Parameters
    ----------
    data_dir : str, optional
        Where to store/find the dataset. Defaults to CONFIG value.
    split_path : str, optional
        Path to split_indices.json. Defaults to 'split_indices.json'.
        If the file exists, the split is loaded (not regenerated).
        Commit this file to GitHub so all team members share it.
    filter_axial : bool, default True
        Filter to axial slices in z-range 60-120.
    label_map : dict or None
        For 4-class (default): None
        For binary (Demented vs NonDemented): {0:1, 1:1, 2:0, 3:1}
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    batch_size : int, optional
        Defaults to CONFIG value (32).
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    meta         : dict with keys:
        loss_weights  : Tensor  — pass to nn.CrossEntropyLoss(weight=...)
        class_names   : list[str]
        n_classes     : int
        train_dir     : str     — path to class folders
        class_to_idx  : dict    — ImageFolder class-to-index mapping
        device        : torch.device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = batch_size or CONFIG["batch_size"]

    if verbose:
        print("=" * 55)
        print("OASIS Preprocessing Pipeline")
        print("=" * 55)
        print(f"Device:        {device}")
        print(f"Batch size:    {batch_size}")
        print(f"Axial filter:  {filter_axial}")
        print(f"Label map:     {'Binary' if label_map else '4-class'}")
        print()

    # 1. Download
    data_dir = download_dataset(data_dir)

    # 2. Find class folders
    train_dir = find_data_dir(data_dir)

    # 3. Load full dataset (no augmentation — just for splitting)
    _, val_tf = get_transforms()
    full_dataset = datasets.ImageFolder(train_dir, transform=val_tf)

    if verbose:
        print(f"\nImageFolder class-to-index: {full_dataset.class_to_idx}")
        print(f"Total images in dataset:    {len(full_dataset)}")

    # 4. Subject-level split
    train_idx, val_idx, test_idx = make_subject_split(
        full_dataset, split_path
    )

    # 5. Build filtered datasets with correct transforms
    train_tf, val_tf = get_transforms()

    base_train = datasets.ImageFolder(train_dir, transform=train_tf)
    base_val   = datasets.ImageFolder(train_dir, transform=val_tf)

    if verbose:
        print(f"\nBuilding datasets"
              f"{' with axial filter' if filter_axial else ''}:")

    train_dataset = OASISSubset(
        base_train, train_idx,
        filter_axial=filter_axial, label_map=label_map
    )
    val_dataset = OASISSubset(
        base_val, val_idx,
        filter_axial=filter_axial, label_map=label_map
    )
    test_dataset = OASISSubset(
        base_val, test_idx,
        filter_axial=filter_axial, label_map=label_map
    )

    # 6. WeightedRandomSampler for class-balanced training batches
    train_labels   = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_counts   = Counter(train_labels)
    sample_weights = [1.0 / label_counts[l] for l in train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True,
    )

    # 7. DataLoaders
    nw = CONFIG["num_workers"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=nw, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=nw, pin_memory=True
    )

    # 8. Class names and loss weights
    if label_map:
        n_classes   = len(set(label_map.values()))
        class_names = ['NonDemented', 'Demented']
    else:
        n_classes   = N_CLASSES
        class_names = CLASS_NAMES

    loss_weights = compute_loss_weights(train_dataset, n_classes, device)

    if verbose:
        print(f"\nFinal dataset sizes:")
        print(f"  Train: {len(train_dataset):>6} images")
        print(f"  Val:   {len(val_dataset):>6} images")
        print(f"  Test:  {len(test_dataset):>6} images")
        print(f"\nLoss weights:")
        for name, w in zip(class_names, loss_weights):
            print(f"  {name:<25} {w:.3f}x")
        print(f"\n✓ Setup complete — ready to train!")
        print("=" * 55)

    meta = {
        'loss_weights':  loss_weights,
        'class_names':   class_names,
        'n_classes':     n_classes,
        'train_dir':     train_dir,
        'class_to_idx':  full_dataset.class_to_idx,
        'device':        device,
    }

    return train_loader, val_loader, test_loader, meta


# ════════════════════════════════════════════════════════════
# SELF-TEST — run: python preprocessing.py
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running self-test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, meta = setup_dataset(
        filter_axial=True,
        device=device,
    )

    # Check one batch from each loader
    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        X, y = next(iter(loader))
        counts = Counter(y.numpy())
        print(f"\n{name} batch: shape={tuple(X.shape)}, "
              f"labels={dict(counts)}")

    print(f"\nClass names:   {meta['class_names']}")
    print(f"N classes:     {meta['n_classes']}")
    print(f"Loss weights:  {meta['loss_weights']}")
    print(f"Class mapping: {meta['class_to_idx']}")
    print("\n✓ Self-test passed!")
