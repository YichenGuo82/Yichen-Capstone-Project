## code recyle backup
## training loop without the incorporation of the labels in the prediction


## Prepare and split the data
tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
train_dataset = TensorDataset(tensor_train)


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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

plt.plot(train_loss_values, label='Train Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


train_latents_stacked = np.vstack(train_latents) 
val_latents_stacked = np.vstack(val_latents) 

print("The shape of train_latent_stacked array is", train_latents_stacked.shape)
print("The shape of val_latent_stacked array is", val_latents_stacked.shape)



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