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


if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


# In[8]:


# Create datasets:
label_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

train_df =  pd.read_csv("../input/train.csv")
val_df = pd.read_csv("../input/val.csv")
test_df = pd.read_csv("../input/test.csv")


# In[9]:


avg_age = train_df['Patient Age'].mean()
std_age = train_df['Patient Age'].std()

def normalize_age(age):
    return (age - avg_age) / std_age


# In[10]:


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
        image = Image.open(img_path).convert("RGB")
        
        image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()
        
        return image, metadata, label


# In[11]:


# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    # Normalize to the mean and standard deviation for each channel in ImageNet Dataset
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize to the mean and standard deviation for each channel in ImageNet Dataset
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# In[12]:


train_ds = ChestXrayDataset(train_df, label_names, "/gpfs/scratch/ms15516/dl4med-project/data/extracted-data/images", train_transform)
val_ds = ChestXrayDataset(val_df, label_names, "/gpfs/scratch/ms15516/dl4med-project/data/extracted-data/images", val_test_transform)
# test_ds = ChestXrayDataset(test_df, "/gpfs/scratch/ms15516/dl4med-project/data/extracted-data/images")

# Create dataloaders:
train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 32)
# test_loader = DataLoader(test_ds, batch_size = 25)

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


# In[13]:


print(f"Number of imges in training dataset: {len(train_ds)}")
print(f"Number of imges in validation dataset: {len(val_ds)}")
# print(f"Number of imges in test dataset: {len(test_ds)}")


# In[14]:


train_df_totals = train_df[label_names].sum()/len(train_df)*100

print(f"Percentage of positive cases for each label in training dataset:")
print(train_df_totals)

prevalence_percent = sum(train_df[label_names].sum())/sum(train_df[label_names].count())*100
print(f"\n\nOverall percentage of positive labels in training dataset: {prevalence_percent:.5f}")


# In[15]:


train_df_weights = (100-train_df_totals) / train_df_totals

train_df_weights


# In[16]:


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


# In[17]:


class_weights_raw = torch.tensor(train_df_weights, dtype=torch.float32)
print(class_weights_raw)
class_weights_log = compute_log_pos_weight(train_df, label_names)
print(class_weights_log)


# In[23]:


# Get subset of training data:
import random

# num_train_subset = 1000
# num_val_subset = 1000

random.seed(123)
subset_indices_train = random.sample(range(len(train_ds)), len(train_ds)//1000)
subset_indices_val = random.sample(range(len(val_ds)), len(val_ds)//1000)
# subset_indices_train = random.sample(range(len(train_ds)), num_train_subset)
# subset_indices_val = random.sample(range(len(val_ds)), num_val_subset)

train_ds_subset = torch.utils.data.Subset(train_ds, subset_indices_train)
train_ds_subset_len = len(train_ds_subset)
print(f"Training Data Subset length: {train_ds_subset_len}")

val_ds_subset = torch.utils.data.Subset(val_ds, subset_indices_val)
val_ds_subset_len = len(val_ds_subset)
print(f"Validation Data Subset length: {val_ds_subset_len}")

# Create subset dataloader
train_subset_loader = DataLoader(train_ds_subset, batch_size = 32, shuffle=True)
val_subset_loader = DataLoader(val_ds_subset, batch_size = 32, shuffle=True)


# In[24]:


import torchvision.models as models

model_resnet = models.resnet50(weights='IMAGENET1K_V2').to(device)
print(model_resnet)


# In[25]:


# Remove output layer / classifier layer from model, but save expected size
original_fc_input_size = model_resnet.fc.in_features
model_resnet.fc = nn.Identity ()

print(model_resnet)
print(" - - - - - - - - - - ")
print(f"original_fc_input_size: {original_fc_input_size}")


# In[26]:


class ResNetConcatHiddenState(nn.Module):
    def __init__(self, resnet_model, original_fc_input_size, num_classes):
        super(ResNetConcatHiddenState, self).__init__()
        self.resnet = resnet_model
        self.classifier = nn.Linear(original_fc_input_size + 3, num_classes)  # +3 for age, sex

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


# In[27]:


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


# In[28]:


# clear cuda memory
import gc

gc.collect()
torch.cuda.empty_cache()


# In[30]:


class_pos_weights = torch.tensor(class_weights_log).to(device)
print(class_pos_weights)


# In[31]:


num_classes = len(label_names)

# # Option 1: Create new model
# model_new = ResNetConcatHiddenState(model_resnet, original_fc_input_size, num_classes).to(device)

# Option 2: Load previously trained model from file to continue training
model_new = ResNetConcatHiddenState(model_resnet, original_fc_input_size, num_classes)
model_new.load_state_dict(torch.load('../output/model_best_wts_log_0428_1.pth'))
model_new.to(device)


# In[36]:


loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weights)
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.001)

# Use scheduler to decrease learning rate by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs = 10

dataloader = {'train': train_loader, 'validate': val_loader}
# dataloader = {'train': train_subset_loader, 'validate': val_subset_loader}

save_path = '../output/model_best_wts_log_0428_1b.pth'

model_resnet_trained, acc_dict, loss_dict = train_model(model_new, dataloader, optimizer, scheduler, loss_fn, save_path, num_epochs, verbose = True)

# - - - Save results to files - - -
torch.save(model_resnet_trained, '../output/model_trained_log_0428_1b.pth')

with open('../output/acc_dict_log_0428_1b.pkl', 'wb') as f:
    pickle.dump(acc_dict, f)

with open('../output/loss_dict_log_0428_1b.pkl', 'wb') as f:
    pickle.dump(loss_dict, f)


# In[37]:


# # Define functions to plot accuracy and loss over training epochs

# def plot_loss(ax, loss, title):
#     x = np.arange(1, len(loss)+1)
#     # plt.figure(dpi=150)
#     ax.plot(x, loss)
#     ax.set_title(title)
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     # plt.show()

# def plot_accuracy(ax, accuracy, title):
#     x = np.arange(1, len(accuracy)+1)
#     # plt.figure(dpi=150)
#     ax.plot(x, accuracy)
#     ax.set_title(title)
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Accuracy')
#     # plt.show()


# In[38]:


# # Visualize the progression of loss and accuracy values during training:
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# plot_loss(axes[0], loss_dict['train'], 'Training Loss')
# plot_accuracy(axes[1], acc_dict['train'], 'Training Accuracy')

# fig.tight_layout()

# fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# plot_loss(axes[0], loss_dict['validate'], 'Validation Loss')
# plot_accuracy(axes[1], acc_dict['validate'], 'Validation Accuracy')

# fig.tight_layout()


# In[ ]:




