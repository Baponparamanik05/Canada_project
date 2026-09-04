# ============================================================
# Sleeping Eye Detection - Improved CNN Training
# ============================================================

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "cnn_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "eye_cnn_fast.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.20
SEED = 42

NUM_WORKERS = 0

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 4. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(8),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ============================================================
# 5. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

full_train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

# Separate dataset with evaluation transforms
full_eval_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

print("Classes:", full_train_dataset.classes)
print("Total training images:", len(full_train_dataset))
print("Total test images:", len(test_dataset))

# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

total_size = len(full_train_dataset)
val_size = int(total_size * VAL_RATIO)
train_size = total_size - val_size

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    total_size,
    generator=generator
).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use training augmentation for training
train_dataset = torch.utils.data.Subset(
    full_train_dataset,
    train_indices
)

# Use evaluation transform for validation
val_dataset = torch.utils.data.Subset(
    full_eval_dataset,
    val_indices
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ============================================================
# 7. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

# Calculate weights from training subset only

train_labels = [
    full_train_dataset.targets[i]
    for i in train_indices
]

class_counts = np.bincount(
    train_labels,
    minlength=len(full_train_dataset.classes)
)

class_weights = len(train_labels) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass counts:")
for name, count in zip(
    full_train_dataset.classes,
    class_counts
):
    print(f"{name}: {count}")

print("\nClass weights:")
print(class_weights)

# ============================================================
# 9. CNN MODEL
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.20)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = EyeCNN().to(device)

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)

# ============================================================
# 10. LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.02
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = 100.0 * correct / total

    average_loss = total_loss / total

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )

# ============================================================
# 12. TRAINING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

best_val_accuracy = 0.0
best_model_state = copy.deepcopy(
    model.state_dict()
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

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
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    # Validation
    val_loss, val_accuracy, _, _ = evaluate(
        model,
        val_loader
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )

    # Save best model using VALIDATION accuracy
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_state,
                "classes": full_train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy
            },
            MODEL_PATH
        )

        print(
            f"  >>> New best validation accuracy: "
            f"{best_val_accuracy:.2f}%"
        )

    scheduler.step(val_accuracy)

# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_model_state)

# ============================================================
# 14. FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST")
print("=" * 60)

test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    model,
    test_loader
)

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n                 Predicted")
print("              closed     open")
print(
    f"Actual closed   {cm[0,0]:4d}      {cm[0,1]:4d}"
)
print(
    f"Actual open     {cm[1,0]:4d}      {cm[1,1]:4d}"
)

# ============================================================
# 16. PER-CLASS ACCURACY
# ============================================================

closed_accuracy = (
    cm[0,0] / cm[0].sum() * 100
    if cm[0].sum() > 0 else 0
)

open_accuracy = (
    cm[1,1] / cm[1].sum() * 100
    if cm[1].sum() > 0 else 0
)

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

print(
    f"Closed Accuracy: {closed_accuracy:.2f}%"
)

print(
    f"Open Accuracy  : {open_accuracy:.2f}%"
)

# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=full_train_dataset.classes,
        digits=4
    )
)

# ============================================================
# 18. SAVE FINAL BEST MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": full_train_dataset.classes,
        "image_size": IMAGE_SIZE,
        "best_val_accuracy": best_val_accuracy,
        "final_test_accuracy": test_accuracy
    },
    MODEL_PATH
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Path:", MODEL_PATH)
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)