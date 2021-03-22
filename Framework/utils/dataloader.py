import os 
import sys
import numpy as np

class dataloader(object):
    def __init__(self,batch_size:int,shuffle:bool) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.shuffle=shuffle
    def __next__(self):
        pass
    def __iter__(self):
        pass