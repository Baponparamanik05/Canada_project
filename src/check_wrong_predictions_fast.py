import os
import shutil
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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wrong_predictions_fast"
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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SLEEPING EYE DETECTION - FAST CNN")
print("WRONG PREDICTION ANALYSIS")
print("=" * 70)
print()

print("Device:", device)
print("Model:", MODEL_PATH)
print("Dataset:", DATA_DIR)
print()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found!")
    print(MODEL_PATH)

    raise SystemExit(1)


# ============================================================
# TRANSFORM
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

class_names = test_dataset.classes

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))
print()


# ============================================================
# FAST CNN MODEL
#
# MUST MATCH THE MODEL USED TO CREATE
# eye_cnn_fast.pth
# ============================================================

class FastEyeCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            # =================================================
            # BLOCK 1
            # =================================================

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


            # =================================================
            # BLOCK 2
            # =================================================

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


            # =================================================
            # BLOCK 3
            # =================================================

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


        # =====================================================
        # CLASSIFIER
        # =====================================================

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


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = FastEyeCNN().to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# CHECKPOINT
# ============================================================

if not isinstance(checkpoint, dict):

    print(
        "ERROR: Unexpected checkpoint format."
    )

    raise SystemExit(1)


if "model_state_dict" not in checkpoint:

    print(
        "ERROR: model_state_dict not found."
    )

    print(
        "Available keys:",
        list(checkpoint.keys())
    )

    raise SystemExit(1)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# MODEL INFORMATION
# ============================================================

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
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )

if "final_test_accuracy" in checkpoint:

    print(
        f"Saved final test accuracy: "
        f"{checkpoint['final_test_accuracy']:.2f}%"
    )

print()


# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

closed_open_dir = os.path.join(
    RESULTS_DIR,
    "actual_closed_predicted_open"
)

open_closed_dir = os.path.join(
    RESULTS_DIR,
    "actual_open_predicted_closed"
)


# ============================================================
# CLEAR OLD FAST RESULTS
# ============================================================

if os.path.exists(RESULTS_DIR):

    print(
        "Removing previous Fast CNN results..."
    )

    shutil.rmtree(
        RESULTS_DIR
    )


os.makedirs(
    closed_open_dir,
    exist_ok=True
)

os.makedirs(
    open_closed_dir,
    exist_ok=True
)


print(
    "Saving wrong predictions to:"
)

print(
    RESULTS_DIR
)

print()


# ============================================================
# TESTING
# ============================================================

print("=" * 70)
print("ANALYZING WRONG PREDICTIONS")
print("=" * 70)
print()


correct = 0
wrong = 0
total = 0

confusion = [
    [0, 0],
    [0, 0]
]


# ============================================================
# PREDICTION LOOP
# ============================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        outputs = model(
            images
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # PROCESS EACH IMAGE
        # ----------------------------------------------------

        for i in range(len(labels)):

            actual = labels[i].item()

            prediction = predictions[i].item()

            confidence = probabilities[
                i,
                prediction
            ].item()


            total += 1


            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            confusion[
                actual
            ][
                prediction
            ] += 1


            # ------------------------------------------------
            # CORRECT
            # ------------------------------------------------

            if actual == prediction:

                correct += 1

                continue


            # ------------------------------------------------
            # WRONG
            # ------------------------------------------------

            wrong += 1


            original_path, _ = test_dataset.samples[
                total - 1
            ]


            filename = os.path.basename(
                original_path
            )


            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            if (
                actual == 0
                and
                prediction == 1
            ):

                destination_dir = (
                    closed_open_dir
                )

                prefix = (
                    "closed_AS_open"
                )

            elif (
                actual == 1
                and
                prediction == 0
            ):

                destination_dir = (
                    open_closed_dir
                )

                prefix = (
                    "open_AS_closed"
                )

            else:

                continue


            # ------------------------------------------------
            # ADD CONFIDENCE TO FILENAME
            # ------------------------------------------------

            confidence_text = (
                f"{confidence * 100:.1f}"
            )

            new_filename = (
                f"{prefix}_"
                f"conf_{confidence_text}_"
                f"{filename}"
            )


            destination_path = os.path.join(
                destination_dir,
                new_filename
            )


            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            shutil.copy2(
                original_path,
                destination_path
            )


# ============================================================
# ACCURACY
# ============================================================

if total > 0:

    accuracy = (
        100.0 *
        correct /
        total
    )

else:

    accuracy = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print()

print(
    f"Total images : {total}"
)

print(
    f"Correct      : {correct}"
)

print(
    f"Wrong        : {wrong}"
)

print(
    f"Accuracy     : {accuracy:.2f}%"
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
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
# PER CLASS ACCURACY
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


print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)
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
# WRONG PREDICTION COUNTS
# ============================================================

closed_as_open = confusion[0][1]

open_as_closed = confusion[1][0]


print("=" * 70)
print("WRONG PREDICTION COUNTS")
print("=" * 70)
print()

print(
    f"Closed predicted as Open : "
    f"{closed_as_open}"
)

print(
    f"Open predicted as Closed : "
    f"{open_as_closed}"
)

print()


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

print("=" * 70)
print("RESULT DIRECTORIES")
print("=" * 70)
print()

print(
    "Closed -> Open:"
)

print(
    closed_open_dir
)

print()

print(
    "Open -> Closed:"
)

print(
    open_closed_dir
)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("FAST CNN WRONG PREDICTION ANALYSIS COMPLETE")
print("=" * 70)