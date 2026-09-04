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
    "eye_cnn_fast.pth"
)

IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SLEEPING EYE DETECTION")
print("FAST CNN EVALUATION")
print("=" * 60)
print()

print("Device:", device)
print("Model:", MODEL_PATH)
print()


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found!")
    print()
    print(MODEL_PATH)

    raise SystemExit(1)


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# DATASET
# ============================================================

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))
print()


# ============================================================
# FAST CNN MODEL
#
# IMPORTANT:
# This architecture must match the architecture used
# when eye_cnn_fast.pth was created.
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()


        # ====================================================
        # FEATURE EXTRACTOR
        # ====================================================

        self.features = nn.Sequential(

            # ------------------------------------------------
            # BLOCK 1
            # ------------------------------------------------

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # ------------------------------------------------
            # BLOCK 2
            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),


            # ------------------------------------------------
            # BLOCK 3
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20)
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                2
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = EyeCNN().to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# CHECKPOINT FORMAT
# ============================================================

if isinstance(checkpoint, dict):

    print(
        "Checkpoint type: dictionary"
    )

    print(
        "Checkpoint keys:",
        list(checkpoint.keys())
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print()
        print(
            "Model state_dict loaded successfully."
        )

    else:

        print()
        print(
            "ERROR: 'model_state_dict' was not found."
        )

        print(
            "Available keys:"
        )

        for key in checkpoint.keys():
            print(
                " -",
                key
            )

        raise SystemExit(1)

else:

    print(
        "Checkpoint is a raw state_dict."
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "Model state_dict loaded successfully."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

if isinstance(checkpoint, dict):

    if "classes" in checkpoint:

        print(
            "Saved classes:",
            checkpoint["classes"]
        )

    if "image_size" in checkpoint:

        print(
            "Saved image size:",
            checkpoint["image_size"]
        )

    if "best_val_accuracy" in checkpoint:

        print(
            "Best validation accuracy:",
            f"{checkpoint['best_val_accuracy']:.2f}%"
        )

    if "final_test_accuracy" in checkpoint:

        print(
            "Saved final test accuracy:",
            f"{checkpoint['final_test_accuracy']:.2f}%"
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


# ============================================================
# CONFUSION MATRIX
#
# Row    = actual
# Column = predicted
#
#          predicted
#          closed open
#
# actual
# closed
# open
# ============================================================

confusion = [
    [0, 0],
    [0, 0]
]


# ============================================================
# PREDICTIONS
# ============================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Overall accuracy
        # ----------------------------------------------------

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        actual_list = labels.cpu().tolist()

        predicted_list = predicted.cpu().tolist()

        for actual, prediction in zip(
            actual_list,
            predicted_list
        ):

            confusion[actual][prediction] += 1


# ============================================================
# OVERALL ACCURACY
# ============================================================

if total > 0:

    accuracy = (
        100.0 *
        correct /
        total
    )

else:

    accuracy = 0.0


wrong = total - correct


# ============================================================
# RESULT
# ============================================================

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

print()

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

print(
    "                    Predicted"
)

print(
    "                 closed     open"
)

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


if closed_total > 0:

    closed_accuracy = (
        100.0 *
        confusion[0][0] /
        closed_total
    )

else:

    closed_accuracy = 0.0


if open_total > 0:

    open_accuracy = (
        100.0 *
        confusion[1][1] /
        open_total
    )

else:

    open_accuracy = 0.0


# ============================================================
# PER-CLASS RESULTS
# ============================================================

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
#
# Calculated manually so sklearn is NOT required.
# ============================================================

# CLOSED

closed_tp = confusion[0][0]

closed_fp = confusion[1][0]

closed_fn = confusion[0][1]


if closed_tp + closed_fp > 0:

    closed_precision = (
        closed_tp /
        (closed_tp + closed_fp)
    )

else:

    closed_precision = 0.0


if closed_tp + closed_fn > 0:

    closed_recall = (
        closed_tp /
        (closed_tp + closed_fn)
    )

else:

    closed_recall = 0.0


if closed_precision + closed_recall > 0:

    closed_f1 = (
        2 *
        closed_precision *
        closed_recall /
        (
            closed_precision +
            closed_recall
        )
    )

else:

    closed_f1 = 0.0


# OPEN

open_tp = confusion[1][1]

open_fp = confusion[0][1]

open_fn = confusion[1][0]


if open_tp + open_fp > 0:

    open_precision = (
        open_tp /
        (open_tp + open_fp)
    )

else:

    open_precision = 0.0


if open_tp + open_fn > 0:

    open_recall = (
        open_tp /
        (open_tp + open_fn)
    )

else:

    open_recall = 0.0


if open_precision + open_recall > 0:

    open_f1 = (
        2 *
        open_precision *
        open_recall /
        (
            open_precision +
            open_recall
        )
    )

else:

    open_f1 = 0.0


# ============================================================
# METRICS
# ============================================================

print("=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)
print()

print(
    "Closed:"
)

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

print(
    "Open:"
)

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


# ============================================================
# FINAL
# ============================================================

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)