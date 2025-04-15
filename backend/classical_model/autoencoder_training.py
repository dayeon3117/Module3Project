import torch
import torch.nn as nn
import torch.optim as optim

def train_autoencoder(model, X_train, epochs, batch_size):
    
    # Convert data to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    
   
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor),
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Initialize model and optimizer
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Adding noise
            inputs = batch[0]
            noisy_inputs = inputs + torch.randn_like(inputs) * 0.1
            
          
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}')
    
    return model