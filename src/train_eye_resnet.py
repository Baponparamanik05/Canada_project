import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
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
    "eye_resnet18.pth"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.0001

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SLEEPING EYE DETECTION - RESNET18 TRAINING")
print("=" * 70)
print()

print("Device:", device)
print("Dataset:", DATA_DIR)
print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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

train_dir = os.path.join(
    DATA_DIR,
    "train"
)

test_dir = os.path.join(
    DATA_DIR,
    "test"
)


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)


test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print("Classes:", train_dataset.class_to_idx)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)

print()


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("Loading ResNet18...")

try:

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    print("Pretrained ResNet18 loaded.")

except Exception as e:

    print()
    print("Could not load pretrained ResNet18.")
    print("Error:", e)
    print()
    print("If this is the first run, an internet connection")
    print("may be required to download the pretrained weights.")
    raise


# ============================================================
# FREEZE MOST OF RESNET
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

number_of_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(
        number_of_features,
        2
    )
)


model = model.to(device)


print("Final layer changed to 2 classes.")
print()


# ============================================================
# CLASS COUNTS
# ============================================================

closed_count = len(
    [
        x for x in train_dataset.samples
        if x[1] == train_dataset.class_to_idx["closed"]
    ]
)

open_count = len(
    [
        x for x in train_dataset.samples
        if x[1] == train_dataset.class_to_idx["open"]
    ]
)


print("Training closed:", closed_count)

print("Training open  :", open_count)

print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_train = closed_count + open_count


closed_weight = (
    total_train /
    (2.0 * closed_count)
)


open_weight = (
    total_train /
    (2.0 * open_count)
)


class_weights = torch.tensor(
    [
        closed_weight,
        open_weight
    ],
    dtype=torch.float32
).to(device)


print("Class weights:")

print(
    "closed:",
    round(closed_weight, 4)
)

print(
    "open  :",
    round(open_weight, 4)
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)
print()


best_accuracy = 0.0

best_model_state = copy.deepcopy(
    model.state_dict()
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item() *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        total
    )


    train_accuracy = (
        100.0 *
        correct /
        total
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    test_correct = 0

    test_total = 0

    test_loss_total = 0.0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            test_loss_total += (
                loss.item() *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            test_total += labels.size(0)


            test_correct += (
                predicted == labels
            ).sum().item()


    test_loss = (
        test_loss_total /
        test_total
    )


    test_accuracy = (
        100.0 *
        test_correct /
        test_total
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_accuracy:.2f}%"
    )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print()

print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "Path:",
    MODEL_PATH
)

print(
    f"Best Test Accuracy: "
    f"{best_accuracy:.2f}%"
)

print()


# ============================================================
# FINAL TEST
# ============================================================

print("=" * 70)
print("FINAL TEST")
print("=" * 70)
print()


model.eval()


correct = 0

total = 0


# Confusion matrix
#
#              Predicted
#              closed   open
#
# Actual closed
# Actual open

confusion = torch.zeros(
    2,
    2,
    dtype=torch.int64
)


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


        for actual, pred in zip(
            labels.cpu(),
            predicted.cpu()
        ):

            confusion[
                actual,
                pred
            ] += 1


# ============================================================
# FINAL ACCURACY
# ============================================================

final_accuracy = (
    100.0 *
    correct /
    total
)


print(
    f"Final Test Accuracy: "
    f"{final_accuracy:.2f}%"
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
    f"Actual closed    "
    f"{confusion[0,0]:6d}    "
    f"{confusion[0,1]:6d}"
)

print(
    f"Actual open      "
    f"{confusion[1,0]:6d}    "
    f"{confusion[1,1]:6d}"
)

print()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

closed_total = (
    confusion[0, 0]
    +
    confusion[0, 1]
)

open_total = (
    confusion[1, 0]
    +
    confusion[1, 1]
)


closed_accuracy = (
    100.0
    *
    confusion[0, 0].item()
    /
    closed_total.item()
)


open_accuracy = (
    100.0
    *
    confusion[1, 1].item()
    /
    open_total.item()
)


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

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)