#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# - - - - - - User input - - - - - 
# Path to folder where images are stored
dataset_dir = "/gpfs/scratch/ms15516/dl4med-project/data/extracted-data/images"
# Set to True to test code on subset of training data, set False to run training
is_use_data_subset = True

# If continuing training:
is_continue_from_checkpoint = True
prev_model_path = ""
# If saving new model: path + file name prefix where model and results will be saved
save_model_path_prefix = "test-output/test-3"


# In[3]:


if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


# In[4]:


# Create datasets:
label_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

train_df =  pd.read_csv("../../input/train.csv")
val_df = pd.read_csv("../../input/val.csv")
test_df = pd.read_csv("../../input/test.csv")


# In[5]:


avg_age = train_df['Patient Age'].mean()
std_age = train_df['Patient Age'].std()

def normalize_age(age):
    return (age - avg_age) / std_age


# In[6]:


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


# In[7]:


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


# In[8]:


train_ds = ChestXrayDataset(train_df, label_names, dataset_dir, train_transform)
val_ds = ChestXrayDataset(val_df, label_names, dataset_dir, val_test_transform)

# Create dataloaders:
train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 32)

# Check that dataloader works as expected:
torch.manual_seed(123)
temp_data = next(iter(train_loader))

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


# In[9]:


print(f"Number of imges in training dataset: {len(train_ds)}")
print(f"Number of imges in validation dataset: {len(val_ds)}")
# print(f"Number of imges in test dataset: {len(test_ds)}")


# In[10]:


train_df_totals = train_df[label_names].sum()/len(train_df)*100

print(f"Percentage of positive cases for each label in training dataset:")
print(train_df_totals)

prevalence_percent = sum(train_df[label_names].sum())/sum(train_df[label_names].count())*100
print(f"\n\nOverall percentage of positive labels in training dataset: {prevalence_percent:.5f}")

prevalence_df = pd.DataFrame(columns=['Condition', 'Percentage of positive cases'], index=None)
prevalence_df['Condition'] = label_names
prevalence_df['Percentage of positive cases'] = train_df_totals.to_numpy()

rounded_prevalence_df = prevalence_df.round({'Percentage of positive cases': 2})
rounded_prevalence_df.style.hide(axis="index")



# In[11]:


train_df_weights = (100-train_df_totals) / train_df_totals

train_df_weights


# In[12]:


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


# In[13]:


class_weights_raw = torch.tensor(train_df_weights, dtype=torch.float32)
print(class_weights_raw)
class_weights_log = compute_log_pos_weight(train_df, label_names)
print(class_weights_log)


# In[14]:


# Get subset of training data:
import random

num_train_subset = 100
num_val_subset = 100

random.seed(123)
subset_indices_train = random.sample(range(len(train_ds)), num_train_subset)
subset_indices_val = random.sample(range(len(val_ds)), num_val_subset)

train_ds_subset = torch.utils.data.Subset(train_ds, subset_indices_train)
train_ds_subset_len = len(train_ds_subset)
print(f"Training Data Subset length: {train_ds_subset_len}")

val_ds_subset = torch.utils.data.Subset(val_ds, subset_indices_val)
val_ds_subset_len = len(val_ds_subset)
print(f"Validation Data Subset length: {val_ds_subset_len}")

# Create subset dataloader
train_subset_loader = DataLoader(train_ds_subset, batch_size = 32, shuffle=True)
val_subset_loader = DataLoader(val_ds_subset, batch_size = 32, shuffle=True)


# In[15]:


# # Calculate approximate parameters required for a convolutional layer to downsample to the expected image size

# input_size = 1024
# pading = 0
# filter_size = 17
# stride = 2

# output_size = np.floor((input_size + (2 * pading) - filter_size) / stride) + 1

# print(output_size)

# input_size = output_size
# pading = 0
# filter_size = 17
# stride = 2

# output_size = np.floor((input_size + (2 * pading) - filter_size) / stride) + 1

# print(output_size)


# In[16]:


# # Calculate approximate parameters required for a convolutional layer to downsample to the expected image size

# input_size = 1024
# pading = 0
# filter_size = 5
# stride = 2

# output_size = np.floor((input_size + (2 * pading) - filter_size) / stride) + 1

# print(output_size)

# input_size = output_size
# pading = 0
# filter_size = 7
# stride = 2

# output_size = np.floor((input_size + (2 * pading) - filter_size) / stride) + 1

# print(output_size)

# input_size = output_size
# pading = 0
# filter_size = 9
# stride = 1

# output_size = np.floor((input_size + (2 * pading) - filter_size) / stride) + 1

# print(output_size)


# In[17]:


import torchvision.models as models

model_resnet = models.resnet50(weights='IMAGENET1K_V2').to(device)
print(model_resnet)


# In[18]:


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


# In[19]:


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


# In[20]:


print(model_resnet)


# In[21]:


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


# In[22]:


import time

# (Based on example code provided in Lab 5:)
def train_model(model, dataloader, optimizer, scheduler, loss_fn, save_path, num_epochs = 1, verbose = False):
    
    phases = ['train','validate']
    acc_dict = {'train':[],'validate':[]}
    exact_acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_exact_acc = 0
    since = time.time()
    
    for i in range(num_epochs):
        print(f"Epoch: {i+1}/{num_epochs}")
        print("- - - - - - - - - - -")
        for p in phases:
            running_exact_correct = 0
            running_correct = 0
            running_loss = 0
            running_total = 0
            if p == 'train':
                model.train()
            else:
                model.eval()
                
            for data in dataloader[p]:
                
                optimizer.zero_grad()
                
                image = data[0].to(device)
                metadata = data[1].to(device)
                label = data[2].to(device)

                # print(f"image shape: {image.size()}")
                
                output = model(image, metadata)
                loss = loss_fn(output, label)

                # print(f"output shape: {output.size()}")
                # print(f"label shape: {label.size()}")
                # print(f"output: {output}")
                # print(f"loss: {loss}")

                preds = torch.round(nn.functional.sigmoid(output))
                # print(f"preds: {preds}, label: {label}")
                
                num_imgs = image.size()[0]
                running_correct += torch.sum(preds == label).item()
                running_loss += loss.item()*num_imgs
                num_classes = label.size()[1]
                running_total += num_imgs*num_classes
                # Exact match accuracy
                running_exact_correct += (preds == label).all(dim=1).sum().item()

                # print(f"running_correct: {running_correct}")
                # print(f"running_total: {running_total}")
                
                if p == 'train':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradient
                    loss.backward()
                    optimizer.step()
                    
            epoch_acc = float(running_correct/running_total)
            epoch_exact_acc = float(running_exact_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            
            if verbose or ((i+1)%10 == 0):
                print('Phase:{}, epoch loss: {:.4f}, Acc: {:.4f}, Exact match acc: {:.4f}'.format(p, epoch_loss, epoch_acc, epoch_exact_acc))
            
            acc_dict[p].append(epoch_acc)
            exact_acc_dict[p].append(epoch_exact_acc)
            loss_dict[p].append(epoch_loss)
            
            if p == 'validate':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    best_val_exact_acc = epoch_exact_acc
                    best_model_wts = model.state_dict()
                    print(f"Saving best trained model to file: {save_path}")
                    torch.save(best_model_wts, save_path)
                if scheduler:
                    scheduler.step(epoch_loss)
         
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f"\n\nBest validation loss: {best_val_loss:.4f}, Acc: {best_val_acc:.4f}, Exact match acc: {best_val_exact_acc:.4f}")

    model.load_state_dict(best_model_wts)
    
    return model, acc_dict, loss_dict


# In[23]:


# clear cuda memory
import gc

gc.collect()
torch.cuda.empty_cache()


# In[24]:


class_pos_weights = torch.tensor(class_weights_log).to(device)
print(class_pos_weights)


# In[25]:


num_classes = len(label_names)

# # Option 1: Create new model
if is_continue_from_checkpoint:
    model_new = ResNetConcatHiddenState(model_resnet, original_fc_input_size, num_classes).to(device)
# Option 2: Load previously trained model from file to continue training
else:
    model_new = ResNetConcatHiddenState(model_resnet, original_fc_input_size, num_classes)
    model_new.load_state_dict(torch.load(prev_model_path))
    model_new.to(device)


# In[26]:


loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weights)
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.001)

# Use scheduler to decrease learning rate by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs = 10

if is_use_data_subset:
    dataloader = {'train': train_subset_loader, 'validate': val_subset_loader}
else:
    dataloader = {'train': train_loader, 'validate': val_loader}

save_path = f"{save_model_path_prefix}_best_weights.pth"

model_resnet_trained, acc_dict, loss_dict = train_model(model_new, dataloader, optimizer, scheduler, loss_fn, save_path, num_epochs, verbose = True)

# - - - Save results to files - - -
torch.save(model_resnet_trained, f"{save_model_path_prefix}_final_model.pth")

with open(f"{save_model_path_prefix}_acc_dict.pkl", 'wb') as f:
    pickle.dump(acc_dict, f)

with open(f"{save_model_path_prefix}_loss_dict.pkl", 'wb') as f:
    pickle.dump(loss_dict, f)

