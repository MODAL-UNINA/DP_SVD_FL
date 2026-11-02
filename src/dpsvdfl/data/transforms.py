"""Image preprocessing (Supplementary Sec. S6.1, Table S1): 224 x 224 RGB inputs with ImageNet normalisation.

These are the CPU (PIL) pipelines used by every experiment of the paper (augmentation backend ``cpu``).
"""

from torchvision import transforms as T

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Random crop covering 80-100% of the (square) image, kept square, resized to 224 x 224.
CROP_SCALE = (0.8, 1.0)
CROP_RATIO = (1.0, 1.0)

GRAYSCALE_DATASETS = ("emnist_balanced",)
RGB_DATASETS = ("cifar10", "cifar100")


def train_transform(dataset_name: str) -> T.Compose:
    """Training preprocessing.

    CIFAR:  random resized crop -> horizontal flip (p = 0.5) -> tensor -> normalise.
    EMNIST: random resized crop -> replicate to 3 channels -> tensor -> normalise (no flip: a flipped character is a
    different or invalid glyph).
    """
    crop = T.RandomResizedCrop(size=IMAGE_SIZE, scale=CROP_SCALE, ratio=CROP_RATIO)
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if dataset_name in RGB_DATASETS:
        return T.Compose([crop, T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalize])
    if dataset_name in GRAYSCALE_DATASETS:
        return T.Compose([crop, T.Grayscale(num_output_channels=3), T.ToTensor(), normalize])
    raise ValueError(f"unknown dataset {dataset_name!r}")


def eval_transform(dataset_name: str) -> T.Compose:
    """Deterministic evaluation preprocessing: resize to exactly 224 x 224 -> (3 channels) -> tensor -> normalise."""
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if dataset_name in RGB_DATASETS:
        return T.Compose([resize, T.ToTensor(), normalize])
    if dataset_name in GRAYSCALE_DATASETS:
        return T.Compose([resize, T.Grayscale(num_output_channels=3), T.ToTensor(), normalize])
    raise ValueError(f"unknown dataset {dataset_name!r}")
