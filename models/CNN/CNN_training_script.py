import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

# -----------------------------
# Load and preprocess CSV
# -----------------------------
train_df =  pd.read_csv('/gpfs/scratch/mmt515/DL_HW/train.csv')
test_df = pd.read_csv('/gpfs/scratch/mmt515/DL_HW/test.csv')
val_df = pd.read_csv('/gpfs/scratch/mmt515/DL_HW/val.csv')

label_names = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']



total_images = len(train_df) + len(val_df) + len(test_df)
print(f"\nImage distribution - Train: {len(train_df)/total_images:.2%}, Val: {len(val_df)/total_images:.2%}, Test: {len(test_df)/total_images:.2%}")


class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels = df[label_names].values

        # Normalize age
        self.df["Patient Age"] = pd.to_numeric(self.df["Patient Age"], errors="coerce").fillna(0).clip(0, 100) / 100.0

        # Encode gender numerically (M=0, F=1, unknown=0.5)
        self.df["Patient Sex"] = self.df["Patient Sex"].map({"M": 0.0, "F": 1.0}).fillna(0.5)

        # Encode View Position numerically: PA=0.0, AP=1.0, unknown=0.5
        self.df["View Position"] = self.df["View Position"].map({"PA": 0.0, "AP": 1.0}).fillna(0.5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_name = row["Image Index"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Use 3 normalized features: age, gender, view_position
        metadata = torch.tensor([
            row["Patient Age"],
            row["Patient Sex"],
            row["View Position"]
        ], dtype=torch.float32)  # shape: (3,)

        label = torch.tensor(self.labels[idx]).float()
        return image, metadata, label


# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------------
# Load Datasets
# -----------------------------
image_dir = "/gpfs/scratch/mmt515/DL_HW/images"

train_dataset = ChestXrayDataset(train_df, image_dir, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, image_dir, transform=val_test_transform)
test_dataset = ChestXrayDataset(test_df, image_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # (batch, 512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512 + 3, num_classes),  # Concatenating 4 metadata features
            nn.Sigmoid()
        )

    def forward(self, x, metadata):
        x = self.features(x)
        x = x.view(x.size(0), -1)              # (batch_size, 512)
        x = torch.cat([x, metadata], dim=1)    # (batch_size, 516)
        x = self.classifier(x)
        return x

# -----------------------------
# Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_names)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




def calculate_accuracies(outputs, labels):
    preds = (outputs > 0.5).float()
    exact_match = (preds == labels).all(dim=1).float().mean().item()
    avg_accuracy = (preds == labels).float().mean().item()
    return exact_match, avg_accuracy

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_exact_match = []
    all_avg_accuracy = []

    for images, metadata, labels in tqdm(loader):
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        exact_match, avg_accuracy = calculate_accuracies(outputs, labels)
        all_exact_match.append(exact_match)
        all_avg_accuracy.append(avg_accuracy)

    return (
        running_loss / len(loader),
        sum(all_exact_match) / len(all_exact_match),
        sum(all_avg_accuracy) / len(all_avg_accuracy)
    )

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_exact_match = []
    all_avg_accuracy = []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            exact_match, avg_accuracy = calculate_accuracies(outputs, labels)
            all_exact_match.append(exact_match)
            all_avg_accuracy.append(avg_accuracy)

    return (
        total_loss / len(loader),
        sum(all_exact_match) / len(all_exact_match),
        sum(all_avg_accuracy) / len(all_avg_accuracy)
    )


train_losses, val_losses = [], []
train_exact_accs, val_exact_accs = [], []
train_avg_accs, val_avg_accs = [], []

best_val_loss = float('inf')

for epoch in range(20):
    train_loss, train_exact, train_avg = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_exact, val_avg = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_exact_accs.append(train_exact)
    val_exact_accs.append(val_exact)
    train_avg_accs.append(train_avg)
    val_avg_accs.append(val_avg)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Exact Match: {train_exact:.4f}, Val Exact Match: {val_exact:.4f}")
    print(f"Train Avg Accuracy: {train_avg:.4f}, Val Avg Accuracy: {val_avg:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_CNN_model.pth")
epochs = range(1, len(train_losses)+1)

plt.figure(figsize=(18, 5))

# Loss Plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Exact Match Accuracy Plot
plt.subplot(1, 3, 2)
plt.plot(epochs, train_exact_accs, label='Train Exact Match Acc')
plt.plot(epochs, val_exact_accs, label='Val Exact Match Acc')
plt.xlabel("Epoch")
plt.ylabel("Exact Match Accuracy")
plt.title("Exact Match Accuracy over Epochs")
plt.legend()

# Average Accuracy Plot
plt.subplot(1, 3, 3)
plt.plot(epochs, train_avg_accs, label='Train Avg Accuracy')
plt.plot(epochs, val_avg_accs, label='Val Avg Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("CNN_training_metrics.png", dpi=300) 

# -----------------------------
# Evaluation Metrics
# -----------------------------
def evaluate_metrics(model, loader, threshold=0.5):
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, metadata, labels in tqdm(loader):
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata)
            y_true.append(labels.numpy())
            y_probs.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_probs = np.vstack(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    metrics = {
        "F1 Score (macro)": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Precision (macro)": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "Exact Match Ratio": (y_true == y_pred).all(axis=1).mean()
    }

    try:
        metrics["AUC-ROC (macro)"] = roc_auc_score(y_true, y_probs, average='macro')
    except ValueError:
        metrics["AUC-ROC (macro)"] = "Undefined (label missing in batch)"

    return metrics, y_true, y_pred


# -----------------------------
# Run test set evaluation
# -----------------------------
# load best model
model.load_state_dict(torch.load('/gpfs/scratch/mmt515/DL_HW/best_CNN_model.pth'))

print("\nEvaluating on test set...\n")
metrics, y_true, y_pred = evaluate_metrics(model, test_loader)

print("\n=== Test Set Performance ===")
for key, val in metrics.items():
    print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

acc_per_label = (y_true == y_pred).mean(axis=0)
label_perf = list(zip(label_names, acc_per_label))
label_perf.sort(key=lambda x: x[1])

print("\nLowest-performing labels:")
for name, acc in label_perf[:5]:
    print(f"{name}: {acc:.3f}")

print("\nHighest-performing labels:")
for name, acc in label_perf[-5:]:
    print(f"{name}: {acc:.3f}")
