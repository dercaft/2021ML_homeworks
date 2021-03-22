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
        ''' Initialize mock data class
            data is: (number, shape_1, ..., shape_n)
        Args:
            number: how many data to generate
            shape: shape of a data sample
            noise: function to generate noise, noise(shape,upper,lower)        
        '''
        self.data=np.zeros( (number,)+shape )
    
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

class Single_poly_mock(Base_mock):
    def __init__(self, rank:int, number:int, noise:object, *args, **kwargs) -> None:
        ''' Initialize mock data class
            data is: (number, y, x)
        Args:
            rank: x order, eg. rank=2 -> y=x^2+x+C where C is a constraint
            number: how many data you want to generate
            noise: noise generator
        '''
        super().__init__(number=number, shape=(2,), noise=noise, args=args, kwargs=kwargs)
        # shape=(2,) (y,x)
        
        
    def __getitem__(self) -> None:
        return super().__getitem__()
    
    def generate(self,range:tuple=(0,1)):
        ''' generate new batch of data
        Returns: new batch
        '''
        pass
    