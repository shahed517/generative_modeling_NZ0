import torch
import torch.nn as nn 

class MAE(nn.Module):
    '''
    Encoder: linaer --> conv --> transformer
    Decoder: transformer --> transpose conv 
    Input shape: (B, 400, 508) --> (B, 2s*200Hz, 508 channels) 
    '''
    
    def __init__(self, n_enc_layers, n_dec_layers):
        super().__init__()
        self.n_enc_layers = n_enc_layers 
        self.n_dec_layers = n_dec_layers 
        
    def forward(self, x): 
        pass 