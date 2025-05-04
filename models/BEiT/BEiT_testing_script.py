import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve
from transformers import BeitModel, BeitImageProcessor

# -----------------------------
# Load and preprocess CSV
# -----------------------------
train_df =  pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('val.csv')

label_names = ['No Finding','Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


avg_age = train_df['Patient Age'].mean()
std_age = train_df['Patient Age'].std()
def normalize_age(age):
    return (age - avg_age) / std_age



image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")

class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, image_processor, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.image_processor = image_processor
        self.labels = df[label_names].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract and normalize metadata
        metadata = self.df.loc[idx, ['Patient Age', 'Patient Sex', 'View Position']].to_list()
        metadata[0] = normalize_age(metadata[0])
        metadata[1] = 1 if metadata[1] == 'M' else 0
        metadata[2] = 1 if metadata[2] == 'AP' else 0
        metadata = torch.tensor(metadata).float()

        # Load and process the image
        img_name = self.df.loc[idx, "Image Index"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        processed = self.image_processor(image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # Ensure this is (C, H, W)
        
        # Ensure labels are the correct type (integer for classification)
        label = torch.tensor(self.labels[idx]).float()

        return pixel_values, metadata, label

# -----------------------------
# Load Datasets
# -----------------------------
image_dir = "./images"

val_dataset = ChestXrayDataset(val_df, image_dir,image_processor)
test_dataset = ChestXrayDataset(test_df, image_dir,image_processor)

val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)



class TransformerTransferLearning(nn.Module):
    def __init__(self, transformer_model, num_classes):
        super().__init__()
        self.transformer = transformer_model
        # Freeze transformer weights
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.metadata_mlp = nn.Sequential(
            nn.Linear(3, 64),  # 3 metadata features: age, sex, view
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine transformer output with processed metadata
        self.classifier = nn.Linear(self.transformer.config.hidden_size + 32, num_classes)

    def forward(self, pixel_values, metadata):
        with torch.no_grad():  # Ensure BEiT stays frozen
            transformer_output = self.transformer(pixel_values)
            hidden_state = transformer_output.last_hidden_state  # [B, L, D]
            avg_embedding = hidden_state.mean(dim=1)  # [B, D]

        # Process metadata (assumes already standardized/encoded)
        metadata_features = self.metadata_mlp(metadata)  # [B, 32]

        # Concatenate transformer + metadata
        combined = torch.cat((avg_embedding, metadata_features), dim=1)  # [B, D+32]

        logits = self.classifier(combined)
        return logits

class TransformerConcatHiddenState(nn.Module):
    def __init__(self, transformer_model, num_classes):
        super(TransformerConcatHiddenState, self).__init__()
        self.transformer = transformer_model
        self.classifier = nn.Linear(self.transformer.config.hidden_size + 3, num_classes)  # +3 for age and sex

    def forward(self, pixel_values, metadata):
        # Get transformer outputs (e.g., CLS token or the final hidden state)
        transformer_output = self.transformer(pixel_values)
        hidden_state = transformer_output.last_hidden_state  # shape: [B, L, D]
        avg_embedding = hidden_state.mean(dim=1)
        

        # Preprocess metadata: standardize age and encode sex
        age = metadata[:, 0].view(-1, 1)  # Make sure age is of shape (batch_size, 1)
        sex = metadata[:, 1].view(-1, 1)  # Make sure sex is of shape (batch_size, 1)
        view = metadata[:, 2].view(-1, 1)


        # Concatenate hidden state with age and sex (assumes sex is already encoded or a scalar)
        combined = torch.cat((avg_embedding, age, sex, view), dim=1)

        # Classification head
        logits = self.classifier(combined)
        return logits

class TransformerSeparateNets(nn.Module):
    def __init__(self, transformer_model, num_classes, metadata_hidden_dim=128):
        super(TransformerSeparateNets, self).__init__()
        self.transformer = transformer_model
        
        # Metadata network: 3 inputs (age, sex, view), outputs hidden vector
        self.metadata_net = nn.Sequential(
            nn.Linear(3, metadata_hidden_dim),
            nn.ReLU(),
            nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
            nn.ReLU()
        )
        
        # Final classifier: transformer CLS + metadata embedding
        self.classifier = nn.Linear(self.transformer.config.hidden_size + metadata_hidden_dim, num_classes)

    def forward(self, pixel_values, metadata):
        # Transformer forward (assumes Huggingface model or compatible)
        transformer_output = self.transformer(pixel_values)
        hidden_state = transformer_output.last_hidden_state  # shape: [B, L, D]
        avg_embedding = hidden_state.mean(dim=1) 

        # Metadata forward
        metadata_embedding = self.metadata_net(metadata)  # (B, H_meta)

        # Concatenate transformer and metadata
        combined = torch.cat((avg_embedding, metadata_embedding), dim=1)  # (B, D + H_meta)

        # Final classification
        logits = self.classifier(combined)
        return logits
# -----------------------------
# Testing
# -----------------------------

model = TransformerConcatHiddenState(beit_model, len(label_names))
model.load_state_dict('TransformerConcatHiddenState_best.pth')
save_path = './checkpoints/TransformerConcatHiddenState/'

def evaluate_metrics(model, loader, threshold=0.5, class_names=None):
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, metadata, labels in tqdm(loader):
            images, metadata = images, metadata
            outputs = model(images, metadata)
            prob = torch.sigmoid(outputs)
            y_true.append(labels.numpy())
            y_probs.append(prob.cpu().numpy())

    y_true = np.vstack(y_true)
    y_probs = np.vstack(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    # Global Metrics
    metrics = {
        "F1 Score (macro)": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Precision (macro)": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "Exact Match Ratio": (y_true == y_pred).all(axis=1).mean()
    }

    try:
        metrics["AUC-ROC (micro)"] = roc_auc_score(y_true, y_probs, average='micro')
        metrics["AUC-ROC (macro)"] = roc_auc_score(y_true, y_probs, average='macro')
        metrics["AUC-ROC (weighted)"] = roc_auc_score(y_true, y_probs, average='weighted')
    except ValueError:
        metrics["AUC-ROC (macro)"] = "Undefined (label missing in batch)"

    # Per-class Metrics
    n_classes = y_true.shape[1]
    class_results = []

    for i in range(n_classes):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        auc = roc_auc_score(y_true[:, i], y_probs[:, i])

        class_name = class_names[i] if class_names else f"Class {i}"

        class_results.append({
            "Class": class_name,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": acc,
            'AUC':auc
        })

    class_metrics_df = pd.DataFrame(class_results)
    # Check if model is predicting all zeros
    all_zero_preds = (y_pred.sum(axis=0) == 0)  # For each class
    if all_zero_preds.any():
        print("\nWARNING: The model is predicting all zeros for the following classes:")
        for i, is_all_zero in enumerate(all_zero_preds):
            if is_all_zero:
                class_name = class_names[i] if class_names else f"Class {i}"
                print(f" - {class_name}")
    else:
        print("\nAll classes have at least some positive predictions.")

    return metrics, class_metrics_df, y_true, y_pred, y_probs


print("\nEvaluating on test set...\n")
metrics, class_metrics_df, y_true, y_pred, y_probs= evaluate_metrics(model, test_loader, threshold=0.5, class_names=label_names)

print("\n=== Test Set Performance ===")
for key, val in metrics.items():
    print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")


print("\nLowest-performing labels (by Accuracy):")
lowest = class_metrics_df.sort_values(by="Accuracy").head(5)
for idx, row in lowest.iterrows():
    print(f"{row['Class']}: {row['Accuracy']:.3f}")

print("\nHighest-performing labels (by Accuracy):")
highest = class_metrics_df.sort_values(by="Accuracy", ascending=False).head(5)
for idx, row in highest.iterrows():
    print(f"{row['Class']}: {row['Accuracy']:.3f}")

# If you also want, you can show the full nice table:
print("\n=== Per-Class Metrics Table ===")
print(class_metrics_df)
# Save predictions and true labels
np.save(f"{save_path}y_true.npy", y_true)
np.save(f"{save_path}y_pred.npy", y_pred)
np.save(f"{save_path}y_probs.npy", y_probs) 

class_metrics_df.to_csv(f'{save_path}class_metric_performance.csv')
colors = plt.get_cmap("tab20").colors
plt.figure(figsize=(10, 8))

for i in range(y_true.shape[1]):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
    auc = roc_auc_score(y_true[:, i], y_probs[:, i])
    label = f"{label_names[i] if label_names else f'Class {i}'} (AUC = {auc:.2f})"
    plt.plot(fpr, tpr, color=colors[i % 10], label=label)

# Diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--', label="Random")

# Labels and legend
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Per Class")
plt.legend(loc="lower right", fontsize='small')
plt.grid(alpha=0.3)

# Save and show
plt.savefig(f"{save_path}roc_curves_per_class.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()
 
# Configuration
n_classes = y_true.shape[1]
cols = 5  
rows = math.ceil(n_classes / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for i in range(n_classes):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[i]
    )
    class_name = label_names[i] if label_names else f"Class {i}"
    axes[i].set_title(class_name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

plt.tight_layout()
plt.savefig(f"{save_path}confusion_matrices_per_class.png", dpi=300)
plt.show()
