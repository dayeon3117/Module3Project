import torch
import torch.nn as nn

def test_model(model, X_test):

    model.eval()
    device = next(model.parameters()).device
    
    # Convert test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    encoder = nn.Sequential(*list(model.children())[:4])
    # Get embeddings
    with torch.no_grad():
        embeddings = encoder(X_test_tensor).cpu().numpy()
    
    return embeddings