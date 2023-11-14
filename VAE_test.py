# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: dsan6600
#     language: python
#     name: python3
# ---

# ## VAE using pytorch

# ### Data exploration

import pandas as pd
import numpy as np
import os
from pyhere import here


# os.chdir(r"D:\Yichen_Guo\YichenG_Code\Yichen-Capstone-Project")
# os.getcwd()

mRNA = pd.read_csv(here("Data/HCC-GU/mRNA_Enzy.csv"),delimiter=";")
miRNA = pd.read_csv(here("Data/HCC-GU/miRNA_Enzy.csv"),delimiter=";")
score_mat =  pd.read_csv(here("Data/HCC-GU/Score_mat_Enzy.csv"),delimiter=";")
patient_labels = pd.read_csv(here("Data/HCC-GU/sample_labels.csv"),delimiter=";") # 1 indicated patient with HCC and CIRR (39), 0 indicated patients with CIRR (25) 

# +
mRNA.head()

## each row is a liver tissue that extract from the 64 patients (marked in the patient_label)

# +
miRNA

## each row is a liver tissue that extract from the 64 patients (marked in the patient_label)

# +
score_mat

## Domain knowledge
## four confiendence level for mRNA-miRNA associations, 1 for expertimentally observed links, 0.75 for highly predicted, 0.5 for moderately predicveted, and 0 for those neirther observed nor predicted
# -

# ## building the autoencoder

# Based on the performace, the potiential of including supervised classification in to the model?
# - To get a better represetntation of the original data?
# - known to work for classification problems

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# +
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out=0.5):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # Batch Normalization layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)  # Batch Normalization layer
        self.dropout = nn.Dropout(drop_out) 
        self.fc_mu = nn.Linear(hidden_dim2, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim2, z_dim)
    
    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))  # Apply Batch Normalization
        h = self.dropout(h)
        h = torch.relu(self.bn2(self.fc2(h)))  # Apply Batch Normalization
        h = self.dropout(h)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, output_dim, drop_out=0.5):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # Batch Normalization layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)  # Batch Normalization layer
        self.dropout = nn.Dropout(drop_out) 
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))  # Apply Batch Normalization
        h = self.dropout(h)
        h = torch.relu(self.bn2(self.fc2(h)))  # Apply Batch Normalization
        h = self.dropout(h)
        recon = self.fc3(h) # using linear due to data type, may use sigmoid if want to use this as classification method
        return recon

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out)
        self.decoder = Decoder(z_dim, hidden_dim1, hidden_dim2, input_dim, drop_out)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD


# -

# ## mRNA test

## Parameters
df_train = mRNA
input_dim = df_train.shape[1]
hidden_dim1 = 1024
hidden_dim2 = 512 
z_dim = 12 
learning_rate = 1e-3
num_epochs = 10000 
drop = 0.3

# ### training loop without the incorporation of the labels in the prediction

# +

## Prepare and split the data
tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
train_dataset_full = TensorDataset(tensor_train)


train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

best_val_loss = float('inf')
patience = 500  # Number of epochs to wait for improvement before stopping
wait = 0


# Model compile

model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out=drop)
model = model.to('cuda') 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

val_loss_values = []
train_loss_values = []
train_latents = []
val_latents = []
train_recon_errors = []
val_recon_errors = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0  # Initialize training loss
    train_recon_error = 0  # Initialize training reconstruction error
    
    # Initialize as empty lists for each epoch
    epoch_train_latents = []  
    
    for batch in train_loader:
        data, = batch
        data = data.to('cuda')
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # Accumulate the training loss
        epoch_train_latents.append(mu.cpu().detach().numpy())  # Append to epoch list
        train_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  # Accumulate training reconstruction error
    
    # Compute the mean latent variables for this epoch
    # epoch_train_latents = np.mean(np.vstack(epoch_train_latents), axis=0)
    # train_latents.append(epoch_train_latents)
    train_latents.extend(epoch_train_latents) 
    
    train_loss /= len(train_loader.dataset)  # Compute the average training loss
    train_loss_values.append(train_loss)
    train_recon_error /= len(train_loader)  # Compute the average training reconstruction error
    train_recon_errors.append(train_recon_error)

    model.eval()
    val_loss = 0
    val_recon_error = 0  # Initialize validation reconstruction error
    
    # Initialize as empty lists for each epoch
    epoch_val_latents = []  
    
    with torch.no_grad():
        for batch in val_loader:
            data, = batch
            data = data.to('cuda')
            recon_batch, mu, log_var = model(data)
            val_loss += loss_function(recon_batch, data, mu, log_var).item()
            epoch_val_latents.append(mu.cpu().detach().numpy())  # Append to epoch list
            val_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  # Accumulate validation reconstruction error

    # Compute the mean latent variables for this epoch
    # epoch_val_latents = np.mean(np.vstack(epoch_val_latents), axis=0)
    # val_latents.append(epoch_val_latents)
    val_latents.extend(epoch_val_latents)

    val_loss /= len(val_loader.dataset)
    val_loss_values.append(val_loss)
    val_recon_error /= len(val_loader)  # Compute the average validation reconstruction error
    val_recon_errors.append(val_recon_error)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0  # Reset wait counter
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping')
            break
# -

plt.plot(train_loss_values, label='Train Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# +
train_latents_stacked = np.vstack(train_latents) 
val_latents_stacked = np.vstack(val_latents) 

print("The shape of train_latent_stacked array is", train_latents_stacked.shape)
print("The shape of val_latent_stacked array is", val_latents_stacked.shape)

# This section adapted from
# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
latent_vars = []
for sample in train_dataset_full:
    input = sample[0].unsqueeze(0).to('cuda')
    model.eval()
    with torch.no_grad():
        encoded= model.encoder(input)
    encoded = encoded[0].flatten().cpu().numpy()
    encoded_sample = {f"LV{i}": enc for i, enc in enumerate(encoded)}
    latent_vars.append(encoded_sample)

latent_vars = pd.DataFrame(latent_vars)
patient_labels.reset_index(inplace=True, drop=True)
latent_vars['labels'] = patient_labels
# +

from torch.utils.tensorboard import SummaryWriter
# Create a figure to hold the subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# PCA
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_latents_stacked)
val_pca = pca.transform(val_latents_stacked)
axs[0].scatter(train_pca[:, 0], train_pca[:, 1], label='Train', alpha=0.5)
axs[0].scatter(val_pca[:, 0], val_pca[:, 1], label='Validation', alpha=0.5)
axs[0].legend()
axs[0].set_title('PCA Reduced Latent Space')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# t-SNE
tsne = TSNE(n_components=2)
train_tsne = tsne.fit_transform(train_latents_stacked)
val_tsne = tsne.fit_transform(val_latents_stacked)
axs[1].scatter(train_tsne[:, 0], train_tsne[:, 1], label='Train', alpha=0.5)
axs[1].scatter(val_tsne[:, 0], val_tsne[:, 1], label='Validation', alpha=0.5)
axs[1].legend()
axs[1].set_title('t-SNE Reduced Latent Space')
axs[1].set_xlabel('Dimension 1')
axs[1].set_ylabel('Dimension 2')

# UMAP
reducer = umap.UMAP()
train_umap = reducer.fit_transform(train_latents_stacked)
val_umap = reducer.transform(val_latents_stacked)
axs[2].scatter(train_umap[:, 0], train_umap[:, 1], label='Train', alpha=0.5)
axs[2].scatter(val_umap[:, 0], val_umap[:, 1], label='Validation', alpha=0.5)
axs[2].legend()
axs[2].set_title('UMAP Reduced Latent Space')
axs[2].set_xlabel('Dimension 1')
axs[2].set_ylabel('Dimension 2')

# Show the plot
plt.tight_layout()
plt.show()
# -

# ### For training with label added (added only for vislualization need)
#
# Trying to use labels for visualization and evaluation of the latent space representations after training (post hoc), then it doesn't modify the VAE model itself or add a supervised component to it. 

# +
## Explaination:
## I have test two different way of loading the dataset in this section due to the discrepency in the labels
## In my current idea of combine the latents spaces (miRNA_latent and mRNA_latent) I was thinking of averaging the rows (which is the trained/val latents vector and labels 
## is appened from the VAE for each sample in the batch)

## Problem 1: My original batch size was set as 16 however due to early stopping in the model, there could be incomplete batch that causing the problems when trying to reshape 
## the latent space in to my desire joint feature space (64* (12+12+12*12)), that why in this segement the batch size is adjusted to 64

## Problem 2: If I choose to average and reshape the latents space vector into my desire shape, how to extract the corresponding latent labels? the labels should be the same across
## the batches (for miRNA and mRNA) but somehow it wasn't 
## Could be due to the random splitiing, the 1st batch of labels for mRNA and miRNA is different (as tested at the end) while it shoudn't be the case as the labels for each batch across
## miRNA and mRNA should be the same, thus introducing split by indice to make sure the mRNA and miRNA has the same split between train and val to make the extraction of label 
## make sense

torch.manual_seed(0)

# Convert data and labels into PyTorch tensors
tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
tensor_labels = torch.tensor(patient_labels.values, dtype=torch.float32) 

# Wrap tensors into a TensorDataset
train_dataset = TensorDataset(tensor_train, tensor_labels)
train_size = int(0.8 * len(train_dataset))

# Generate indices for training and validation sets
indices = torch.randperm(len(train_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create samplers using the generated indices
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# DataLoader using the samplers
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler)

best_val_loss = float('inf')
patience = 500  # Number of epochs to wait for improvement before stopping
wait = 0

# +
## The original way of data loading and spliting

# Prepare and split the data
tensor_labels = torch.tensor(patient_labels.values, dtype=torch.float32)
tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
train_dataset_full = TensorDataset(tensor_train, tensor_labels)


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

best_val_loss = float('inf')
patience = 500  # Number of epochs to wait for improvement before stopping
wait = 0

# +
# Model compile

model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out=drop)
model = model.to('cuda') 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

val_loss_values = []
train_loss_values = []
train_latents = []
val_latents = []
train_labels = []  # Storing labels for training set
val_labels = []  # Storing labels for validation set
train_recon_errors = []
val_recon_errors = []

# We'll now store the average data from each batch instead of individual samples.
train_data_avg_batch = []  
train_sample_recon_errors = []  # Storing average reconstruction error for each batch

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0  
    train_recon_error = 0  
    
    for batch in train_loader:
        data, labels = batch  # Now you're also getting the labels
        data = data.to('cuda')
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Store the average sample from each batch
        avg_data = data.mean(0).cpu().detach().numpy()
        train_data_avg_batch.append(avg_data)
        
        train_latents.extend(mu.cpu().detach().numpy())  # Directly extending the train_latents list
        train_labels.extend(labels.cpu().detach().numpy())  # Directly extending the train_labels list
        
        # Get the average reconstruction error for the batch and store it
        train_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  
        batch_recon_errors = F.mse_loss(recon_batch, data, reduction='none').mean(1).cpu().detach().numpy()
        avg_batch_recon_error = batch_recon_errors.mean()
        train_sample_recon_errors.append(avg_batch_recon_error)
    
    train_loss /= len(train_loader.dataset)
    train_loss_values.append(train_loss)
    train_recon_error /= len(train_loader)
    train_recon_errors.append(train_recon_error)

    model.eval()
    val_loss = 0
    val_recon_error = 0  
    
    with torch.no_grad():
        for batch in val_loader:
            data, labels = batch  # Now you're also getting the labels
            data = data.to('cuda')
            recon_batch, mu, log_var = model(data)
            val_loss += loss_function(recon_batch, data, mu, log_var).item()
            val_latents.extend(mu.cpu().detach().numpy())  # Directly extending the val_latents list
            val_labels.extend(labels.cpu().detach().numpy())  # Directly extending the val_labels list
            val_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  
    
    val_loss /= len(val_loader.dataset)
    val_loss_values.append(val_loss)
    val_recon_error /= len(val_loader)
    val_recon_errors.append(val_recon_error)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0  # Reset wait counter
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping')
            break

train_data_avg_batch = np.array(train_data_avg_batch)
train_sample_recon_errors = np.array(train_sample_recon_errors)

# +
# After the loop, if you wish to convert the lists to arrays:
mRNA_train_latents = np.array(train_latents)
mRNA_train_labels = np.array(train_labels)
mRNA_val_latents = np.array(val_latents)
mRNA_val_labels = np.array(val_labels)

print("The shape of val_latent array is", mRNA_val_latents.shape)
print("The shape of val_label array is", mRNA_val_labels.shape)
print("The shape of train_latents array is", mRNA_train_latents.shape)
print("The shape of train label array is", mRNA_train_labels.shape)

# +
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# PCA
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_latents)
val_pca = pca.transform(val_latents)
scatter = axs[0].scatter(train_pca[:, 0], train_pca[:, 1], c=train_labels, alpha=0.5)
axs[0].scatter(val_pca[:, 0], val_pca[:, 1], c=val_labels, alpha=0.5)
legend1 = axs[0].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[0].add_artist(legend1)
axs[0].set_title('PCA Reduced Latent Space')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# t-SNE
tsne = TSNE(n_components=2)
all_latents = np.vstack([train_latents, val_latents])
all_labels = np.hstack([train_labels.reshape(-1), val_labels.reshape(-1)])
all_tsne = tsne.fit_transform(all_latents)
scatter = axs[1].scatter(all_tsne[:len(train_labels), 0], all_tsne[:len(train_labels), 1], c=train_labels, alpha=0.5)
axs[1].scatter(all_tsne[len(train_labels):, 0], all_tsne[len(train_labels):, 1], c=val_labels, alpha=0.5)
legend2 = axs[1].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[1].add_artist(legend2)
axs[1].set_title('t-SNE Reduced Latent Space')
axs[1].set_xlabel('Dimension 1')
axs[1].set_ylabel('Dimension 2')

# UMAP
reducer = umap.UMAP()
train_umap = reducer.fit_transform(train_latents)
val_umap = reducer.transform(val_latents)
scatter = axs[2].scatter(train_umap[:, 0], train_umap[:, 1], c=train_labels, alpha=0.5)
axs[2].scatter(val_umap[:, 0], val_umap[:, 1], c=val_labels, alpha=0.5)
legend3 = axs[2].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[2].add_artist(legend3)
axs[2].set_title('UMAP Reduced Latent Space')
axs[2].set_xlabel('Dimension 1')
axs[2].set_ylabel('Dimension 2')

# Show the plot
plt.tight_layout()
plt.show()
# -

# ### Using SVM to evaluate the performance of the VAE code

# +
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_latents2 = np.vstack(train_latents)

# Initialize the SVM classifier
clf = SVC()

# Train the SVM classifier using training latent space and labels
clf.fit(train_latents2, train_labels)

# Predict on the validation data
val_predictions = clf.predict(val_latents)

# Calculate the performance metrics
accuracy = accuracy_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions)
recall = recall_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions)

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# -

# ###  Reconstruction Error Contribution
#
# Train a linear model (like a linear regression) to predict the reconstruction error of each sample based on its feature values. Features that have higher coefficients in this model can be interpreted as having a greater influence on the reconstruction error.
#
# In the training loop, for each batch, we store the average sample in train_data_avg_batch, and the reconstruction error for that average sample in train_sample_recon_errors. Each entry in train_sample_recon_errors corresponds to the reconstruction error for the average sample of each batch. We can then use train_data_avg_batch as our feature matrix and train_sample_recon_errors as our target variable to train a linear regression model. By examining the coefficients of the linear model, we can infer the importance of each feature in terms of its contribution to the reconstruction error.
#

# +
from sklearn.linear_model import LinearRegression

# Using the original data (tensor_train) and train_recon_errors to train the linear model
reg = LinearRegression().fit(train_data_avg_batch, train_sample_recon_errors)

# Get the coefficients
coefficients = reg.coef_

# Rank features by importance
ranked_features_by_error_contribution = np.argsort(np.abs(coefficients))[::-1]

# Display top N influential features
N = 12
top_N_features = ranked_features_by_error_contribution[:N]
print("Top N features based on their influence on reconstruction error:", top_N_features)
linselected_mRNA_names = [df_train.columns.tolist()[i] for i in top_N_features]
print(linselected_mRNA_names)

# -

# ## miRNA test

## Parameters
df_train = miRNA
input_dim = df_train.shape[1]
hidden_dim1 = 1024
hidden_dim2 = 512 
z_dim = 12
learning_rate = 1e-3
num_epochs = 10000 
drop = 0.3

# +
# Convert data and labels into PyTorch tensors
tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
tensor_labels = torch.tensor(patient_labels.values, dtype=torch.float32) 

# Wrap tensors into a TensorDataset
train_dataset = TensorDataset(tensor_train, tensor_labels)

# Create samplers using the previously generated indices
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# DataLoader using the samplers
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler)


best_val_loss = float('inf')
patience = 500  # Number of epochs to wait for improvement before stopping
wait = 0

# +
# Model compile

model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim, drop_out=drop)
model = model.to('cuda') 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

val_loss_values = []
train_loss_values = []
train_latents = []
val_latents = []
train_labels = []  # Storing labels for training set
val_labels = []  # Storing labels for validation set
train_recon_errors = []
val_recon_errors = []

# We'll now store the average data from each batch instead of individual samples.
train_data_avg_batch = []  
train_sample_recon_errors = []  # Storing average reconstruction error for each batch

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0  
    train_recon_error = 0  
    
    for batch in train_loader:
        data, labels = batch  # Now you're also getting the labels
        data = data.to('cuda')
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Store the average sample from each batch
        avg_data = data.mean(0).cpu().detach().numpy()
        train_data_avg_batch.append(avg_data)
        
        train_latents.extend(mu.cpu().detach().numpy())  # Directly extending the train_latents list
        train_labels.extend(labels.cpu().detach().numpy())  # Directly extending the train_labels list
        
        # Get the average reconstruction error for the batch and store it
        train_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  
        batch_recon_errors = F.mse_loss(recon_batch, data, reduction='none').mean(1).cpu().detach().numpy()
        avg_batch_recon_error = batch_recon_errors.mean()
        train_sample_recon_errors.append(avg_batch_recon_error)
    
    train_loss /= len(train_loader.dataset)
    train_loss_values.append(train_loss)
    train_recon_error /= len(train_loader)
    train_recon_errors.append(train_recon_error)

    model.eval()
    val_loss = 0
    val_recon_error = 0  
    
    with torch.no_grad():
        for batch in val_loader:
            data, labels = batch  # Now you're also getting the labels
            data = data.to('cuda')
            recon_batch, mu, log_var = model(data)
            val_loss += loss_function(recon_batch, data, mu, log_var).item()
            val_latents.extend(mu.cpu().detach().numpy())  # Directly extending the val_latents list
            val_labels.extend(labels.cpu().detach().numpy())  # Directly extending the val_labels list
            val_recon_error += F.mse_loss(recon_batch, data, reduction='mean').item()  
    
    val_loss /= len(val_loader.dataset)
    val_loss_values.append(val_loss)
    val_recon_error /= len(val_loader)
    val_recon_errors.append(val_recon_error)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0  # Reset wait counter
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping')
            break

train_data_avg_batch = np.array(train_data_avg_batch)
train_sample_recon_errors = np.array(train_sample_recon_errors)

# +
# plt.plot(train_loss_values, label='Train Loss')
# plt.plot(val_loss_values, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# +
# After the loop, if you wish to convert the lists to arrays:
miRNA_train_latents = np.array(train_latents)
miRNA_train_labels = np.array(train_labels)
miRNA_val_latents = np.array(val_latents)
miRNA_val_labels = np.array(val_labels)

print("The shape of val_latent array is", miRNA_val_latents.shape)
print("The shape of val_label array is", miRNA_val_labels.shape)
print("The shape of train_latents array is", miRNA_train_latents.shape)
print("The shape of train label array is", miRNA_train_labels.shape)

# +
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# PCA
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_latents)
val_pca = pca.transform(val_latents)
scatter = axs[0].scatter(train_pca[:, 0], train_pca[:, 1], c=train_labels, alpha=0.5)
axs[0].scatter(val_pca[:, 0], val_pca[:, 1], c=val_labels, alpha=0.5)
legend1 = axs[0].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[0].add_artist(legend1)
axs[0].set_title('PCA Reduced Latent Space')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# t-SNE
tsne = TSNE(n_components=2)
all_latents = np.vstack([train_latents, val_latents])
all_labels = np.hstack([train_labels.reshape(-1), val_labels.reshape(-1)])
all_tsne = tsne.fit_transform(all_latents)
scatter = axs[1].scatter(all_tsne[:len(train_labels), 0], all_tsne[:len(train_labels), 1], c=train_labels, alpha=0.5)
axs[1].scatter(all_tsne[len(train_labels):, 0], all_tsne[len(train_labels):, 1], c=val_labels, alpha=0.5)
legend2 = axs[1].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[1].add_artist(legend2)
axs[1].set_title('t-SNE Reduced Latent Space')
axs[1].set_xlabel('Dimension 1')
axs[1].set_ylabel('Dimension 2')

# UMAP
reducer = umap.UMAP()
train_umap = reducer.fit_transform(train_latents)
val_umap = reducer.transform(val_latents)
scatter = axs[2].scatter(train_umap[:, 0], train_umap[:, 1], c=train_labels, alpha=0.5)
axs[2].scatter(val_umap[:, 0], val_umap[:, 1], c=val_labels, alpha=0.5)
legend3 = axs[2].legend(handles=scatter.legend_elements()[0], labels=['0', '1'], title="Classes")
axs[2].add_artist(legend3)
axs[2].set_title('UMAP Reduced Latent Space')
axs[2].set_xlabel('Dimension 1')
axs[2].set_ylabel('Dimension 2')

# Show the plot
plt.tight_layout()
plt.show()
# -

# ### PCA on the original 

# +
pca = PCA(n_components=2)

# Apply PCA to the miRNA dataset
pca_miRNA = pca.fit_transform(miRNA)

# Apply PCA to the mRNA dataset
pca_mRNA = pca.fit_transform(mRNA)

# Create a figure to hold the subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plot the PCA results for miRNA
axs[0].scatter(pca_miRNA[:, 0], pca_miRNA[:, 1], alpha=0.5)
axs[0].set_title('PCA Reduced miRNA Data')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')

# Plot the PCA results for mRNA
axs[1].scatter(pca_mRNA[:, 0], pca_mRNA[:, 1], alpha=0.5)
axs[1].set_title('PCA Reduced mRNA Data')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')

# Show the plot
plt.tight_layout()
plt.show()
# -

# ### Lasso regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# +
X_train, X_val, y_train, y_val = train_test_split(df_train, patient_labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Fit Lasso model
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Reconstruction Error
lasso_recon = lasso.predict(X_train_scaled)
lasso_recon_error = mean_squared_error(y_train, lasso_recon)

# Classification (assuming a threshold of 0.5 to classify)
lasso_class = (lasso.predict(X_val_scaled) > 0.5).astype(int)
lasso_acc = accuracy_score(y_val, lasso_class)
lasso_f1 = f1_score(y_val, lasso_class)

# +

# Reconstruction Error (assuming you have a function get_reconstruction that gets the reconstruction of X from the VAE)
# Print errors
print(f'VAE Train Reconstruction Error: {train_recon_error}')
print(f'VAE Validation Reconstruction Error: {val_recon_error}')
print(f'Lasso Prediction Error: {lasso_recon_error}')
# -

# ### Joint Feature space

# - Create a combined dataset where each entry is a concatenation of the VAE latent representation of a specific mRNA and miRNA.
# - Introduce interaction terms. For every pair (latent_mRNA_i, latent_miRNA_j), introduce a new feature that is their product. This serves as an interaction term capturing combined effects of that specific mRNA and miRNA latent feature.
#
# problem: the discrepancy between the number of rows for miRNA and mRNA latent space. (storing the latent representations for each batch in each epoch. This leads to multiple latent representations for the same data point across different epochs.)
#
# solution:  first average the latent spaces for each patient across all epochs. This will condense the information and provide one average latent representation for each patient. After averaging, concatenate the latent vectors side-by-side and compute interaction terms
#
# For each epoch, for each batch, you are appending latent vectors (mu values) from the VAE for each sample in the batch to train_latents (and their corresponding labels to train_labels). Since your size is 16, for every batch processed, we're adding 16 latent vectors to train_latents.
#
# The number of batches processed per epoch can be calculated based on the size of training set and your batch size. Given that original data consists of 64 patients, and using a batch size of 16, there are 64/16 = 4 batches per epoch. . So the total number of latent vectors stored in train_latents will be N * 4 * 16 = 64N.

# +
# Concatenate train and val latents
combined_mRNA_latents_all = np.vstack((mRNA_train_latents, mRNA_val_latents))
combined_miRNA_latents_all = np.vstack((miRNA_train_latents, miRNA_val_latents))

# Averaging across batches (now considering 64 patients)
averaged_mRNA_latents = np.mean(combined_mRNA_latents_all.reshape(-1, 64, 12), axis=0)
averaged_miRNA_latents = np.mean(combined_miRNA_latents_all.reshape(-1, 64, 12), axis=0)

# Combine the averaged latent representations
combined_latents = np.hstack((averaged_mRNA_latents, averaged_miRNA_latents))

# Compute interaction terms
interaction_terms = np.einsum('ij,ik->ijk', averaged_mRNA_latents, averaged_miRNA_latents).reshape(64, -1)

# Joint dataset
joint_dataset = np.hstack((combined_latents, interaction_terms))

# +
trimmed_mRNA_latents = mRNA_train_latents[:batches_processed_mRNA * 16, :]
trimmed_miRNA_latents = miRNA_train_latents[:batches_processed_miRNA * 16, :]

# Now you can reshape and average
averaged_mRNA_latents = np.mean(trimmed_mRNA_latents.reshape(batches_processed_mRNA, 16, -1), axis=0)
averaged_miRNA_latents = np.mean(trimmed_miRNA_latents.reshape(batches_processed_miRNA, 16, -1), axis=0)

# Rest of the code remains the same


# Rest of the code remains the same

# Combine the latent representations
combined_latents = np.hstack((averaged_mRNA_latents, averaged_miRNA_latents))

# Compute interaction terms
interaction_terms = np.einsum('ij,ik->ijk', averaged_mRNA_latents, averaged_miRNA_latents).reshape(64, -1)

# joint dataset
joint_dataset = np.hstack((combined_latents, interaction_terms))

# -

print(mRNA_train_latents.shape)
print(miRNA_train_latents.shape)


np.array_equal(miRNA_train_labels[:64], mRNA_train_labels[:64])
