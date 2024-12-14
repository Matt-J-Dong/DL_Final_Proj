def main():
    device = get_device()

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 5
    input_dim = 8450
    output_dim = 256

    # Load full dataset first
    data_path = "/scratch/DL24FA"
    full_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=1  # Set to 1 temporarily to access individual samples
    )

    # Extract 1% subset
    dataset_size = len(full_loader.dataset)  # Assuming the dataset is attached to the DataLoader
    subset_size = max(1, int(dataset_size * 0.01))  # 1% of the dataset
    subset_indices = random.sample(range(dataset_size), subset_size)  # Randomly sample indices
    subset = Subset(full_loader.dataset, subset_indices)  # Create a subset

    # Create a DataLoader for the subset
    train_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True
    )

    # Initialize model
    model = SingleLinearModel(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # We'll train the model to predict a zero vector of dimension 256 for each sample, just for testing
    # In reality, you would have a proper target tensor.
    
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states.to(device)  # states: [B,T,C,H,W]
            actions = batch.actions.to(device) # actions: [B,T-1,action_dim]

            B,T,C,H,W = states.shape
            # Flatten the initial state (e.g., states[:,0]) into 8450-dim vector
            # Assuming states[:,0] has shape [B, C, H, W], we flatten it
            init_state = states[:,0].view(B, -1)
            assert init_state.size(1) == input_dim, f"Expected init_state to have {input_dim} features, got {init_state.size(1)}"

            # Forward pass
            preds = model(init_state)  # [B, 256]

            # Create dummy target: zero tensor
            targets = torch.zeros(B, output_dim, device=device)

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    main()
