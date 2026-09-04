import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"D:\SleepingEyeDetection"

DATA_DIR = os.path.join(
    BASE_DIR,
    "cnn_dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "eye_cnn.pth"
)

IMAGE_SIZE = 64
BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("SLEEPING EYE DETECTION")
print("ORIGINAL CNN EVALUATION")
print("=" * 60)
print()

print("Device:", device)
print("Model:", MODEL_PATH)
print("Dataset:", DATA_DIR)
print()


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))
print()


# ============================================================
# EXACT ARCHITECTURE FROM eye_cnn.pth
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # features.0
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            # features.1
            nn.BatchNorm2d(32),

            # features.2
            nn.ReLU(inplace=True),

            # features.3
            nn.MaxPool2d(2),

            # features.4
            nn.Dropout2d(0.1),

            # features.5
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            # features.6
            nn.BatchNorm2d(64),

            # features.7
            nn.ReLU(inplace=True),

            # features.8
            nn.MaxPool2d(2),

            # features.9
            nn.Dropout2d(0.1),

            # features.10
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            # features.11
            nn.BatchNorm2d(128),

            # features.12
            nn.ReLU(inplace=True),

            # features.13
            nn.MaxPool2d(2),

            # features.14
            nn.Dropout2d(0.1),

            # features.15
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            # features.16
            nn.BatchNorm2d(256),

            # features.17
            nn.ReLU(inplace=True),

            # features.18
            nn.MaxPool2d(2),

            # features.19
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.classifier = nn.Sequential(

            # classifier.0
            nn.Flatten(),

            # classifier.1
            nn.Linear(
                256,
                64
            ),

            # classifier.2
            nn.ReLU(inplace=True),

            # classifier.3
            nn.Dropout(0.3),

            # classifier.4
            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = EyeCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

print(
    "Checkpoint type:",
    type(checkpoint).__name__
)


# Original eye_cnn.pth is an OrderedDict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


# ============================================================
# LOAD STATE DICT
# ============================================================

model.load_state_dict(
    state_dict
)

model.eval()

print()
print("Model loaded successfully.")
print()


# ============================================================
# TESTING
# ============================================================

print("=" * 60)
print("TESTING")
print("=" * 60)
print()


correct = 0
total = 0

confusion = [
    [0, 0],
    [0, 0]
]

class_names = test_dataset.classes


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        for actual, prediction in zip(
            labels.cpu().numpy(),
            predicted.cpu().numpy()
        ):

            confusion[actual][prediction] += 1


# ============================================================
# RESULT
# ============================================================

wrong = total - correct

accuracy = (
    100.0 *
    correct /
    total
)


print("=" * 60)
print("RESULT")
print("=" * 60)
print()

print(
    f"Correct predictions: "
    f"{correct}/{total}"
)

print(
    f"Wrong predictions: "
    f"{wrong}/{total}"
)

print(
    f"Test Accuracy: "
    f"{accuracy:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print()

print("                    Predicted")
print("                 closed     open")

print(
    f"Actual closed   "
    f"{confusion[0][0]:6d} "
    f"{confusion[0][1]:8d}"
)

print(
    f"Actual open     "
    f"{confusion[1][0]:6d} "
    f"{confusion[1][1]:8d}"
)

print()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

closed_total = (
    confusion[0][0] +
    confusion[0][1]
)

open_total = (
    confusion[1][0] +
    confusion[1][1]
)


closed_accuracy = (
    100.0 *
    confusion[0][0] /
    closed_total
)

open_accuracy = (
    100.0 *
    confusion[1][1] /
    open_total
)


print("=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)
print()

print(
    f"Closed Accuracy: "
    f"{closed_accuracy:.2f}%"
)

print(
    f"Open Accuracy  : "
    f"{open_accuracy:.2f}%"
)

print()


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

closed_tp = confusion[0][0]
closed_fp = confusion[1][0]
closed_fn = confusion[0][1]

open_tp = confusion[1][1]
open_fp = confusion[0][1]
open_fn = confusion[1][0]


def metrics(tp, fp, fn):

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1 = 0.0

    return precision, recall, f1


closed_precision, closed_recall, closed_f1 = metrics(
    closed_tp,
    closed_fp,
    closed_fn
)

open_precision, open_recall, open_f1 = metrics(
    open_tp,
    open_fp,
    open_fn
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print("=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)
print()

print("Closed:")

print(
    f"  Precision: "
    f"{closed_precision * 100:.2f}%"
)

print(
    f"  Recall   : "
    f"{closed_recall * 100:.2f}%"
)

print(
    f"  F1 Score : "
    f"{closed_f1 * 100:.2f}%"
)

print()

print("Open:")

print(
    f"  Precision: "
    f"{open_precision * 100:.2f}%"
)

print(
    f"  Recall   : "
    f"{open_recall * 100:.2f}%"
)

print(
    f"  F1 Score : "
    f"{open_f1 * 100:.2f}%"
)

print()

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)