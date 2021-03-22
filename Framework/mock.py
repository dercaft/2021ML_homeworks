import os
import sys
import numpy as np

class Noiser:
    def __init__(self) -> None:
        raise NotImplementedError
    def generate(self):
        raise NotImplementedError

class Base_mock:
    data=None
    
    def __init__(self, number:int, shape:tuple, noise:object, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def __getitem__(self):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError
    
    def __next__(self):
        raise NotImplementedError
    
    def generate(self):
        raise NotImplementedError
    
    def refresh(self) -> None:
        self.generate()

    def get_all(self):
        raise NotImplementedError

class Poly_mock(Base_mock):
    def __init__(self, number:int, shape:tuple, noise:object, *args, **kwargs) -> None:
        ''' initialize mock data class
        Args:
            number: how many data to generate
            shape: shape of a data sample
            noise: function to generate noise, noise(shape,upper,lower)
        Returns:
        
        '''
        super().__init__()
        
        self.data=np.zeros( (number,)+shape )
        
    def __getitem__(self) -> None:
        return super().__getitem__()
    
    def generate(self):
        pass
    