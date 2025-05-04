import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from transformers import BeitModel, BeitImageProcessor,get_linear_schedule_with_warmup
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
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])

# -----------------------------
# Load Datasets
# -----------------------------
image_dir = "./images"

train_dataset = ChestXrayDataset(train_df, image_dir, image_processor, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, image_dir,image_processor)
test_dataset = ChestXrayDataset(test_df, image_dir,image_processor)

train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)


# -----------------------------
# Models
# -----------------------------


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
# Training Setup
# -----------------------------
checkpoint_dir = './checkpoints/TransformerConcatHiddenState'
os.makedirs(checkpoint_dir, exist_ok=True)
model = TransformerConcatHiddenState(beit_model, len(label_names))

def compute_log_weights_from_df(df, label_names, epsilon=1e-6):
    # Get number of samples
    total_samples = len(df)

    # Count positive values per label
    pos_counts = df[label_names].sum(axis=0)  # pandas Series
    freq = pos_counts / (total_samples + epsilon)

    # Compute log-weighted inverse frequencies
    weights = 1.0 / np.log(1.02 + freq)

    return torch.tensor(weights.values, dtype=torch.float32)

loss_weights = compute_log_weights_from_df(train_df, label_names)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0
    exact_match_correct = 0
    total_samples = 0
    total_avg_acc = 0.0

    for images, metadata, labels in tqdm(loader):
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()

        # Exact match accuracy
        exact_match_correct += (preds == labels).all(dim=1).sum().item()

        # Average accuracy per sample
        avg_acc_batch = ((preds == labels).float().mean(dim=1)).sum().item()
        total_avg_acc += avg_acc_batch

        total_samples += labels.size(0)

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'loss': loss,  # or whatever metric you want to track
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    avg_loss = running_loss / len(loader)
    exact_match_accuracy = exact_match_correct / total_samples
    average_accuracy = total_avg_acc / total_samples

    return avg_loss, exact_match_accuracy, average_accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    exact_match_correct = 0
    total_samples = 0
    total_avg_acc = 0.0

    with torch.no_grad():
        for images, metadata, labels in loader:
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Exact match accuracy
            exact_match_correct += (preds == labels).all(dim=1).sum().item()

            # Average accuracy per sample
            avg_acc_batch = ((preds == labels).float().mean(dim=1)).sum().item()
            total_avg_acc += avg_acc_batch

            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    exact_match_accuracy = exact_match_correct / total_samples
    average_accuracy = total_avg_acc / total_samples

    return avg_loss, exact_match_accuracy, average_accuracy


# -----------------------------
# Train the model
# -----------------------------


total_epochs = 20
num_training_steps = total_epochs * len(train_loader)

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                             num_warmup_steps=int(0.1 * num_training_steps), 
                                             num_training_steps=num_training_steps)

# if loading from checkpoint \


def save_training_metadata_txt(model, save_dir, params):
    # Get current date and time
    saved_on = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Prepare the metadata
    lines = []
    lines.append(f"Saved on: {saved_on}\n")
    lines.append(f"Model structure:\n{str(model)}\n")
    
    # Append the parameters
    for key, value in params.items():
        lines.append(f"{key}: {value}")
    
    # Save to a txt file
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "training_metadata.txt"), "w") as f:
        f.write("\n".join(lines))

params = {
    "output_label_size": len(label_names),
    "num_epochs": total_epochs,
    "optimizer": "Adam",
    "learning_rate": 3e-4,
    "scheduler": "linear_warmup",
    "warmup_steps": int(0.1 * num_training_steps),
    "batch_size": 60,
    "loss_function": "BCEWithLogitsLoss",
    "loss_weights": loss_weights.tolist(),
    "model_name": model.__class__.__name__
}

# Define your save directory

# Call the function after training or at checkpoint
save_training_metadata_txt(model, checkpoint_dir, params)

train_losses, val_losses = [], []
train_exact_accs, val_exact_accs = [], []
train_avg_accs, val_avg_accs = [], []
best_val_loss= float('inf')

save_path = 'TransformerConcatHiddenState_best.pth'
for epoch in range(total_epochs):  # You can increase this
    train_loss,train_exact,avg_train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss,val_exact,avg_val_acc= evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_exact_accs.append(train_exact)
    val_exact_accs.append(val_exact)
    train_avg_accs.append(avg_train_acc)
    val_avg_accs.append(avg_val_acc)

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Epoch {epoch+1} - Train Exact Match Accuracy: {train_exact:.4f}, Val Exact Match Accuracy: {val_exact:.4f}")
    print(f"Epoch {epoch+1} - Train Avg Accuracy: {avg_train_acc:.4f}, Val Avg Accuracy: {avg_val_acc:.4f}")
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(18, 5))

    # Loss Plot - Saves after each epoch in case time limit is reached
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
    plt.savefig(f"{checkpoint_dir}/TransformerConcatHiddenState.png", dpi=300) 


