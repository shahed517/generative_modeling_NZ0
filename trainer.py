import torch, time, os, sys, pickle 
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader
from dataset import NWBDataset, create_dataloader
from MAE_model import MAE 

class BrainToSpeech_Trainer: 
    def __init__(self, args):
        self.args = args 
        
        
        # TODO: initialize model, dataloader (train, val, trainval), optimizer
        
        # TODO: train with MSE, L1, contrastive 
        
        # TODO: initialize logger with wandb 
        pass 
    
    def train(self):
        pass 
    
    def validation(self):
        pass 
