#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import itertools
from skimage import io
import math
import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle
import pandas as pd
from PIL import Image, ImageOps


# In[13]:


# Path to folder where images are stored
dataset_dir = "/gpfs/scratch/ms15516/dl4med-project/data/extracted-data/images"


# In[14]:


if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


# In[15]:


# Create datasets:
label_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

train_df =  pd.read_csv("../../input/train.csv")
val_df = pd.read_csv("../../input/val.csv")
test_df = pd.read_csv("../../input/test.csv")


# In[16]:


import torchvision.models as models

model_resnet = models.resnet50(weights='IMAGENET1K_V2').to(device)
print(model_resnet)


# In[17]:


class customNormalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # print(f"original mean: {torch.mean(torch.flatten(x))}")
        # print(f"original std: {torch.std(torch.flatten(x))}")
        normalized_x = x*self.std + self.mean
        # print(f"new mean: {torch.mean(torch.flatten(normalized_x))}")
        # print(f"new std: {torch.std(torch.flatten(normalized_x))}")
        return normalized_x


# In[18]:


# Remove output layer / classifier layer from model, but save expected size
original_fc_input_size = model_resnet.fc.in_features
model_resnet.fc = nn.Identity ()

print(f"original_fc_input_size: {original_fc_input_size}")

# Add additional convolutional layers at the beginning to change input size from [1, 1024, 1024] to [3, 224, 224]
input_layer = model_resnet.conv1
# Also use a custom normalization layer to match expected input distributions of pre-trained ResNet
# For simplicity, average across RGB channels, since the values are similar
imagenet_mean = np.mean([0.485, 0.456, 0.406])
imagenet_std = np.mean([0.229, 0.224, 0.225])
normalization_layer = customNormalization(imagenet_mean, imagenet_std)

new_input_layer = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0),
    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(32, 16, kernel_size=7, stride=2, padding=0),
    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(16, input_layer.in_channels, kernel_size=9, stride=1, padding=0),
    nn.BatchNorm2d(input_layer.in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    normalization_layer,
    input_layer
)

model_resnet.conv1 = new_input_layer.to(device)


# In[19]:


class ResNetConcatHiddenState(nn.Module):
    def __init__(self, resnet_model, original_fc_input_size, num_classes):
        super(ResNetConcatHiddenState, self).__init__()
        self.resnet = resnet_model
        self.classifier = nn.Linear(original_fc_input_size + 3, num_classes)  # +3 for age, sex, view

    def forward(self, image, metadata):
        # Get resnet output (ie. final hidden state)
        resnet_output = self.resnet(image)
        # print(f"hidden state shape: {resnet_output.size()}")
        
        # Add metadata
        age = metadata[:, 0].unsqueeze(dim=1)  # Make sure age is of shape (batch_size, 1)
        sex = metadata[:, 1].unsqueeze(dim=1) # Make sure sex is of shape (batch_size, 1)
        view = metadata[:, 2].unsqueeze(dim=1) # Make sure view is of shape (batch_size, 1)

        # Concatenate hidden state with age and sex (assumes sex is already encoded or a scalar)
        combined = torch.cat((resnet_output, age, sex, view), dim=1)
        # print(f"hidden state + metadata shape: {combined.size()}")

        # Classification head
        logits = self.classifier(combined)
        
        return logits


# In[20]:


# Load previously trained model from file
num_classes = len(label_names)
model_resnet_trained = ResNetConcatHiddenState(model_resnet, original_fc_input_size, num_classes)
model_resnet_trained.load_state_dict(torch.load('../../output/model_resnet50_log_weights_conv_input.pth'))
model_resnet_trained.to(device)

# Load data from the pickle files
with open('../../output/acc_dict_resnet50_log_weights_conv_input.pkl', 'rb') as file:
    acc_dict = pickle.load(file)

with open('../../output/loss_dict_resnet50_log_weights_conv_input.pkl', 'rb') as file:
    loss_dict = pickle.load(file)

print(acc_dict)
print(loss_dict)


# In[21]:


avg_age = train_df['Patient Age'].mean()
std_age = train_df['Patient Age'].std()

def normalize_age(age):
    return (age - avg_age) / std_age


# In[22]:


class ChestXrayDataset(Dataset):
    def __init__(self, df, label_names, image_dir, transform):

        self.df = df.reset_index(drop=True)
        # print(self.df.head())
        # print(self.df.columns)
        
        self.image_dir = image_dir
        self.transform = transform
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
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()
        
        return image, metadata, label


# In[23]:


# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

val_test_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])


# In[24]:


train_ds = ChestXrayDataset(train_df, label_names, dataset_dir, train_transform)
val_ds = ChestXrayDataset(val_df, label_names, dataset_dir, val_test_transform)
test_ds = ChestXrayDataset(test_df, label_names, dataset_dir, val_test_transform)

# Create dataloaders:
train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 32)
test_loader = DataLoader(test_ds, batch_size = 32)

# Check that dataloader works as expected:
torch.manual_seed(123)
temp_data = next(iter(test_loader))

temp_img = temp_data[0]
temp_metadata = temp_data[1]
temp_label = temp_data[2]

print(" - - - - Sample Image Pixel Values - - - - ")
print(temp_img)
print(" - - - - - - - - - ")
print(f"temp_img shape: {temp_img.shape}")
plt.imshow(temp_img[0,0,:,:], cmap='gray')
print(f"temp_metadata: {temp_metadata[0]}")
print(f"temp_label shape: {temp_label.shape}")
print(f"temp_label: {temp_label[0]}")


# In[25]:


print(f"Number of imges in training dataset: {len(train_ds)}")
print(f"Number of imges in validation dataset: {len(val_ds)}")
print(f"Number of imges in test dataset: {len(test_ds)}")


# In[26]:


train_df_totals = train_df[label_names].sum()/len(train_df)*100

print(f"Percentage of positive cases in training dataset:")
train_df_totals


# In[27]:


test_df_totals = test_df[label_names].sum()/len(test_df)*100

print(f"Percentage of positive cases in testing dataset:")
test_df_totals


# In[28]:


# Define functions to plot accuracy and loss over training epochs

def plot_loss(ax, loss, title):
    x = np.arange(1, len(loss)+1)
    # plt.figure(dpi=150)
    ax.plot(x, loss)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # plt.show()

def plot_accuracy(ax, accuracy, title):
    x = np.arange(1, len(accuracy)+1)
    # plt.figure(dpi=150)
    ax.plot(x, accuracy)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    # plt.show()


# In[29]:


# Visualize the progression of loss and accuracy values during training:
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_loss(axes[0], loss_dict['train'], 'Training Loss')
plot_accuracy(axes[1], acc_dict['train'], 'Training Accuracy')

fig.tight_layout()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

plot_loss(axes[0], loss_dict['validate'], 'Validation Loss')
plot_accuracy(axes[1], acc_dict['validate'], 'Validation Accuracy')

fig.tight_layout()


# In[30]:


# Create modified version of train_model function, which includes sending back all predicted labels (needed to calcualte AUC metric later)
def evaluate_model(model, dataloader,loss_fn, phase = 'validate'):
    
    model.eval()
    
    running_correct = 0
    running_loss = 0
    running_total = 0
    batch_counter = 0
    total_batches = len(dataloader[phase])
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():  # Disable gradient tracking
        for data in dataloader[phase]:
            print(f"Batch {batch_counter}/{total_batches}")
            batch_counter += 1
            image = data[0].to(device)
            metadata = data[1].to(device)
            label = data[2].to(device)
            output = model(image, metadata)
            loss = loss_fn(output, label)
            probs = nn.functional.sigmoid(output)
            preds = torch.round(probs)
            num_imgs = image.size()[0]
            running_correct += torch.sum(preds ==label).item()
            running_loss += loss.item()*num_imgs
            num_classes = label.size()[1]
            running_total += num_imgs*num_classes
    
            # print(f"probs shape: {probs.size()}")
            # print(f"label shape: {label.size()}")
            # print(f"preds shape: {preds.size()}")

            all_labels.append(label.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
        
    accuracy = float(running_correct/running_total)
    loss = float(running_loss/running_total)

    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    print(f"all_labels shape: {all_labels.size()}")
    print(f"all_probs shape: {all_probs.size()}")
    print(f"all_preds shape: {all_preds.size()}")
    
    return all_preds, all_labels, all_probs, accuracy, loss


# In[31]:


train_df_weights = (100-train_df_totals) / train_df_totals

print(train_df_weights)


# In[32]:


def compute_log_pos_weight(train_df, label_names, max_cap=20.0):
    num_samples = len(train_df)
    # print(f"num_samples: {num_samples}")
    num_labels = len(label_names)
    # print(f"num_labels: {num_labels}")
    label_counts = train_df[label_names].sum()
    # print(f"label_counts: {label_counts}")
    # Avoid divide-by-zero and smooth
    weights = (num_samples - label_counts) / (label_counts + 1e-6)
    # print(f"weights: {weights}")
    # Apply log scale and cap
    weights = np.minimum(np.log1p(weights), max_cap)
    # print(f"weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


# In[33]:


class_weights_raw = torch.tensor(train_df_weights, dtype=torch.float32)
print(class_weights_raw)
class_weights_log = compute_log_pos_weight(train_df, label_names)
print(class_weights_log)

class_pos_weights = torch.tensor(class_weights_log).to(device)


# In[34]:


# Predict labels for validation dataset using model:
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weights)
eval_dataloader = {'test': val_loader}

val_preds_list, val_labels_list, val_probs_list, val_accuracy, val_loss = evaluate_model(model_resnet_trained, eval_dataloader, loss_fn, 'test')


# In[35]:


# - - - Save results to files - - -
with open('../../output/inference_results/val_preds_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(val_preds_list, f)

with open('../../output/inference_results/val_labels_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(val_labels_list, f)

with open('../../output/inference_results/val_probs_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(val_probs_list, f)

with open('../../output/inference_results/val_accuracy_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(val_accuracy, f)

with open('../../output/inference_results/val_loss_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(val_loss, f)


# In[36]:


# Load data from the pickle files
with open('../../output/inference_results/val_preds_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    val_preds_list = pickle.load(f)

with open('../../output/inference_results/val_labels_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    val_labels_list = pickle.load(f)

with open('../../output/inference_results/val_probs_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    val_probs_list = pickle.load(f)

with open('../../output/inference_results/val_accuracy_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    val_accuracy = pickle.load(f)

with open('../../output/inference_results/val_loss_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    val_loss = pickle.load(f)


# In[37]:


# Predict labels for testing dataset using model:
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weights)
eval_dataloader = {'test': test_loader}

test_preds_list, test_labels_list, test_probs_list, test_accuracy, test_loss = evaluate_model(model_resnet_trained, eval_dataloader, loss_fn, 'test')


# In[38]:


# - - - Save results to files - - -
with open('../../output/inference_results/test_preds_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(test_preds_list, f)

with open('../../output/inference_results/test_labels_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(test_labels_list, f)

with open('../../output/inference_results/test_probs_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(test_probs_list, f)

with open('../../output/inference_results/test_accuracy_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(test_accuracy, f)

with open('../../output/inference_results/test_loss_resnet50_log_weights_conv_input.pkl', 'wb') as f:
    pickle.dump(test_loss, f)


# In[39]:


# Load data from the pickle files
with open('../../output/inference_results/test_preds_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    test_preds_list = pickle.load(f)

with open('../../output/inference_results/test_labels_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    test_labels_list = pickle.load(f)

with open('../../output/inference_results/test_probs_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    test_probs_list = pickle.load(f)

with open('../../output/inference_results/test_accuracy_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    test_accuracy = pickle.load(f)

with open('../../output/inference_results/test_loss_resnet50_log_weights_conv_input.pkl', 'rb') as f:
    test_loss = pickle.load(f)


# In[40]:


results_dict = {}

for idx, label in enumerate(label_names):
    results_dict[f"predicted_{label}"] = test_preds_list[:,idx].bool()
    results_dict[f"label_{label}"] = test_labels_list[:,idx].bool()
    results_dict[f"prob_{label}"] = test_probs_list[:,idx]

results_df = pd.DataFrame(results_dict)
print(results_df)


# In[41]:


from sklearn.metrics import recall_score, precision_score, f1_score

y_true = test_labels_list.numpy().astype(int)
print(y_true)

y_pred = test_preds_list.numpy().astype(int)
print(y_pred)


# In[42]:


# Compute recall: tp / (tp + fn)
recall_per_class = recall_score(y_true, y_pred, average=None)  # or 'macro' or 'micro', 'samples', 'weighted'
# print("Recall score:", recall)

# Compute precision: tp / (tp + fp)
precision_per_class = precision_score(y_true, y_pred, average=None)

# Compute F-1 score: tp / (tp + fp)
f1score_per_class = f1_score(y_true, y_pred, average=None)

df_class_scores = pd.DataFrame(columns=label_names, index=['Recall','Precision','F-1 Score'])

df_class_scores.loc['Recall'] = recall_per_class
df_class_scores.loc['Precision'] = precision_per_class
df_class_scores.loc['F-1 Score'] = f1score_per_class

# Display dataframe:
df_class_scores


# In[56]:


# Macro
recall_macro = recall_score(y_true, y_pred, average='macro')
precision_macro = precision_score(y_true, y_pred, average='macro')
f1score_macro = f1_score(y_true, y_pred, average='macro')

# Micro
recall_micro = recall_score(y_true, y_pred, average='micro')
precision_micro = precision_score(y_true, y_pred, average='micro')
f1score_micro = f1_score(y_true, y_pred, average='micro')

data = {'Macro' : [recall_macro, precision_macro, f1score_macro], 'Micro': [recall_micro, precision_micro, f1score_micro]}
df_avg_scores = pd.DataFrame(data, index=['Recall','Precision','F-1 Score'])
df_avg_scores.round({'Macro': 4, 'Micro': 4, 'Weighted': 4})


# In[44]:


test_df_totals = test_df[label_names].sum()/len(test_df)*100

print(f"Percentage of positive cases in testing dataset:")
test_df_totals


# In[45]:


columns = ["predicted_" + s for s in label_names]

preds_df_totals = results_df[columns].sum()/len(results_df)*100

print(f"Percentage of predicted positive cases for the testing dataset:")
preds_df_totals


# In[46]:


# exact match ratio
exact_correct = np.all(y_true == y_pred, axis=1)
print(f"Exact match percentage: {sum(exact_correct)/len(test_ds)*100:.2f}%")


# In[47]:


# Calculate optimal operating point thresholds based on validation dataset:

from sklearn.metrics import precision_recall_curve, f1_score

num_labels = len(label_names)

probs = val_probs_list.numpy()
labels = val_labels_list.numpy().astype(int)

# Before threshold tuning:
preds = (probs >= 0.5).astype(int)
f1_macro = f1_score(labels, preds, average='macro')
f1_micro = f1_score(labels, preds, average='micro')
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")

# Find optimal threshold per label using PR curve + F1
optimal_thresholds = []
for i in range(num_labels):
    precision, recall, thresholds = precision_recall_curve(labels[:, i], probs[:, i])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_thresholds.append(best_threshold)

optimal_thresholds = np.array(optimal_thresholds)
print("Optimal thresholds per label:", optimal_thresholds)

# Apply thresholds for updated predictions
preds = (probs >= optimal_thresholds).astype(int)

# Evaluate F1 score after threshold tuning
f1_macro = f1_score(labels, preds, average='macro')
f1_micro = f1_score(labels, preds, average='micro')
print(f"F1 Macro (tuned): {f1_macro:.4f}")
print(f"F1 Micro (tuned): {f1_micro:.4f}")


# In[48]:


def predict_with_thresholds(probs, thresholds):
    adjusted_predictions = (probs >= torch.tensor(thresholds, device=probs.device)).float()
    
    return adjusted_predictions


# In[49]:


# Recalculate results for test dataset
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
print(f"F1 Macro (original): {f1_macro:.4f}")
print(f"F1 Micro (original): {f1_micro:.4f}")

test_preds_list_tuned = predict_with_thresholds(test_probs_list, optimal_thresholds)
y_pred_tuned = test_preds_list_tuned.numpy().astype(int)

f1_macro = f1_score(y_true, y_pred_tuned, average='macro')
f1_micro = f1_score(y_true, y_pred_tuned, average='micro')
print(f"F1 Macro (tuned): {f1_macro:.4f}")
print(f"F1 Micro (tuned): {f1_micro:.4f}")


# In[50]:


# Compute recall: tp / (tp + fn)
recall_per_class = recall_score(y_true, y_pred_tuned, average=None)  # or 'macro' or 'micro', 'samples', 'weighted'
# print("Recall score:", recall)

# Compute precision: tp / (tp + fp)
precision_per_class = precision_score(y_true, y_pred_tuned, average=None)

# Compute F-1 score: tp / (tp + fp)
f1score_per_class = f1_score(y_true, y_pred_tuned, average=None)

df_class_scores_tuned = pd.DataFrame(columns=label_names, index=['Recall','Precision','F-1 Score'])

df_class_scores_tuned.loc['Recall'] = recall_per_class
df_class_scores_tuned.loc['Precision'] = precision_per_class
df_class_scores_tuned.loc['F-1 Score'] = f1score_per_class

# Display dataframe:
df_class_scores_tuned


# In[55]:


# Macro
recall_macro = recall_score(y_true, y_pred_tuned, average='macro')
precision_macro = precision_score(y_true, y_pred_tuned, average='macro')
f1score_macro = f1_score(y_true, y_pred_tuned, average='macro')

# Micro
recall_micro = recall_score(y_true, y_pred_tuned, average='micro')
precision_micro = precision_score(y_true, y_pred_tuned, average='micro')
f1score_micro = f1_score(y_true, y_pred_tuned, average='micro')

data = {'Macro' : [recall_macro, precision_macro, f1score_macro], 'Micro': [recall_micro, precision_micro, f1score_micro]}
df_avg_scores_tuned = pd.DataFrame(data, index=['Recall','Precision','F-1 Score'])
df_avg_scores_tuned.round({'Macro': 4, 'Micro': 4, 'Weighted': 4})


# In[52]:


# Calculate AUC-ROC:

from sklearn.metrics import roc_auc_score

# y_true: [batch_size, num_labels], binary (0 or 1)
# y_pred: [batch_size, num_labels], probabilities from model (sigmoid output)

y_probs = test_probs_list

macro_auc = roc_auc_score(y_true, y_probs, average='macro')
micro_auc = roc_auc_score(y_true, y_probs, average='micro')
weighted_auc = roc_auc_score(y_true, y_probs, average='weighted')

print("Macro AUC:", macro_auc)
print("Micro AUC:", micro_auc)
print("Weighted AUC:", weighted_auc)


# In[53]:


# Plot ROC curves:

from sklearn.metrics import roc_curve, auc

# Plotting setup
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
cmap=plt.get_cmap('tab20')

roc_auc_list = []

for i in range(num_labels):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {roc_auc:.2f})", color=cmap(i))
    roc_auc_list.append(roc_auc)

# Chance line
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
ax.set_aspect('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Class (Multilabel)')

plt.legend(loc='lower right')
handles, labels = plt.gca().get_legend_handles_labels()
order = np.argsort(roc_auc_list)[::-1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])


plt.grid(True)
plt.tight_layout()
plt.savefig('../../output/inference_results/model-7-log-weights-roc-auc-1.png', dpi=300)
plt.show()


# In[54]:


from sklearn.metrics import precision_recall_curve, average_precision_score

# Create a figure
plt.figure(figsize=(10, 8))
cmap=plt.get_cmap('tab20')
ap_score_list = []

for i in range(num_labels):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
    ap_score = average_precision_score(y_true[:, i], y_probs[:, i])
    ap_score_list.append(ap_score)
    
    plt.plot(recall, precision, label=f'{label_names[i]} (AP = {ap_score:.2f})', color=cmap(i))

# Plot settings
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Per Label')

plt.legend(loc='best')
handles, labels = plt.gca().get_legend_handles_labels()
order = np.argsort(ap_score_list)[::-1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.grid(True)
plt.savefig('../../output/inference_results/model-7-log-weights-prc-1.png', dpi=300)
plt.show()


# In[ ]:




