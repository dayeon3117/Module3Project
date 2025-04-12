import torch.nn as nn

def denoising_autoencoder(input_dim, encoding_dim):

    
    encoder = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, encoding_dim),
        nn.ReLU()
    )
    
   
    decoder = nn.Sequential(
        nn.Linear(encoding_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim),
        nn.Sigmoid()
    )
    
    
    model = nn.Sequential(encoder, decoder)
    
    return model