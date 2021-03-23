import os 
import sys
import numpy as np

class base_dataloader(object):
    def __init__(self, dataset, batch_size:int,shuffle:bool) -> None:
        super().__init__()
        self.dataset=dataset
        self.length=dataset.__len__()
        self.batch_size=batch_size
        self.b_num=self.length//self.batch_size
        self.b_left=self.length%self.batch_size
        self.b_index=0
        
        self.shuffle=shuffle
        self.i_list=np.array([x for x in range(dataset.__len__())],dtype=np.int)
        if(self.shuffle):
            np.random.shuffle(self.i_list)
            
        self.index=0
    def __next__(self):
        if(not self.index < self.length):
            if(self.shuffle):
                np.random.shuffle(self.i_list)
            self.index=0
            self.b_index=0
            raise StopIteration
        raise NotImplementedError
    def __iter__(self):
        return self

class once_dataloader(base_dataloader):
    ''' suitable dataset type: load all data into memory once w.r.t LAD datatset
    '''
    def __init__(self, dataset, batch_size: int, shuffle: bool) -> None:
        super().__init__(dataset, batch_size, shuffle)
        