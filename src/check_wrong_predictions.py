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
    "eye_cnn.pth"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wrong_predictions"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SLEEPING EYE DETECTION - WRONG PREDICTION ANALYSIS")
print("=" * 70)
print()

print("Device:", device)
print("Model:", MODEL_PATH)
print()


# ============================================================
# TEST TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# LOAD TEST DATASET
# ============================================================

test_dir = os.path.join(
    DATA_DIR,
    "test"
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
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
# EXACT SAME CNN AS TRAINING
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # BLOCK 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # BLOCK 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.10),


            # BLOCK 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),


            # BLOCK 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20)
        )


        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

model = EyeCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# CREATE RESULT DIRECTORIES
# ============================================================

if os.path.exists(RESULT_DIR):

    shutil.rmtree(RESULT_DIR)


closed_to_open_dir = os.path.join(
    RESULT_DIR,
    "actual_closed_predicted_open"
)

open_to_closed_dir = os.path.join(
    RESULT_DIR,
    "actual_open_predicted_closed"
)


os.makedirs(
    closed_to_open_dir,
    exist_ok=True
)

os.makedirs(
    open_to_closed_dir,
    exist_ok=True
)


# ============================================================
# CLASS INDEX
# ============================================================

closed_index = test_dataset.class_to_idx["closed"]

open_index = test_dataset.class_to_idx["open"]


# ============================================================
# TESTING
# ============================================================

correct = 0
total = 0

wrong_closed_to_open = 0
wrong_open_to_closed = 0

sample_index = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        for i in range(len(labels)):

            actual = labels[i].item()

            predicted = predictions[i].item()


            original_path, _ = (
                test_dataset.samples[sample_index]
            )

            sample_index += 1


            # ------------------------------------------------
            # WRONG PREDICTION
            # ------------------------------------------------

            if actual != predicted:

                confidence = probabilities[
                    i,
                    predicted
                ].item()

                filename = os.path.basename(
                    original_path
                )


                # CLOSED -> OPEN

                if (
                    actual == closed_index
                    and
                    predicted == open_index
                ):

                    wrong_closed_to_open += 1

                    destination = os.path.join(
                        closed_to_open_dir,

                        f"{wrong_closed_to_open:04d}_"
                        f"conf_{confidence:.2f}_"
                        f"{filename}"
                    )


                # OPEN -> CLOSED

                elif (
                    actual == open_index
                    and
                    predicted == closed_index
                ):

                    wrong_open_to_closed += 1

                    destination = os.path.join(
                        open_to_closed_dir,

                        f"{wrong_open_to_closed:04d}_"
                        f"conf_{confidence:.2f}_"
                        f"{filename}"
                    )


                shutil.copy2(
                    original_path,
                    destination
                )


# ============================================================
# RESULTS
# ============================================================

accuracy = (
    100.0 *
    correct /
    total
)


print("=" * 70)
print("RESULT")
print("=" * 70)
print()

print(
    "Correct predictions:",
    correct
)

print(
    "Wrong predictions:",
    total - correct
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print()

print(
    "Closed -> Open errors:",
    wrong_closed_to_open
)

print(
    "Open -> Closed errors:",
    wrong_open_to_closed
)

print()

print("=" * 70)
print("WRONG IMAGES SAVED")
print("=" * 70)
print()

print(
    RESULT_DIR
)

print()

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)