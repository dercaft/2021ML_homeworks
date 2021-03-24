import os
import sys
from types import LambdaType
import numpy as np

class Noise:
    def __init__(self,shape:tuple,ranger:tuple, *args, **kwargs) -> None:
        ''' Noise generater, based on np.random packages
            Because the base range is (0,1), I set lower as bias, (upper - lower) as scale
        Args:
            shape: data shape
            ranger: noise range, for example, (0,1)
        '''
        self.lower=np.min(ranger)
        self.upper=np.max(ranger)
        self.mean =np.mean(ranger)
        self.scale=np.abs(ranger[1]-ranger[0])
        self.shape=shape
    def generate(self,bs:int=1):
        ''' Noise generater'''
        raise NotImplementedError
    
class Gauss_noise(Noise):
    def __init__(self, shape: tuple, ranger: tuple, *args, **kwargs) -> None:
        super().__init__(shape, ranger, *args, **kwargs)
    def generate(self,bs:int=1):
        noise=np.random.normal(loc=self.mean,scale=self.scale//2,size=tuple([bs])+self.shape)
        return noise
    
class Uniform_noise(Noise):
    def __init__(self, shape: tuple, ranger: tuple, *args, **kwargs) -> None:
        super().__init__(shape, ranger, *args, **kwargs)
    def generate(self,bs:int=1):
        noise=np.random.uniform(self.lower,self.upper,size=tuple([bs])+self.shape)
        return noise

class Base_mock:    
    def __init__(self, number:int, shape:tuple, noise_generator:Noise=None, *args, **kwargs) -> None:
        ''' Initialize mock data class
            data is: (number, shape_1, ..., shape_n)
        Args:
            number: how many data to generate
            shape: shape of a data sample
            noise: function to generate noise, noise(shape,upper,lower)        
        '''
        self.data=np.zeros( (number,)+shape )
        self.number=number
        self.noise_generater=noise_generator
        self.noise=np.zeros( (number,)+shape )
    def generate(self):
        ''' return mock data '''
        return self.data+self.noise
    
    def generate_no_noise(self):
        return self.data
    
    def refresh_noise(self) -> None:
        assert self.noise_generater
        self.noise=self.noise_generater.generate(self.number)

class Single_base_mock(Base_mock):
    def __init__(self, ranger:tuple, number: int,noise_generator: Noise=None,func=None, *args, **kwargs) -> None:
        ''' Single x single y mock , eg. (y,x) -> (1,2)
        
        '''
        assert func
        super().__init__(number=number, shape=(2,), noise_generator=noise_generator, args=args, kwargs=kwargs)
        
        self.func=func
        if(not noise_generator):
            self.noise_generater=Uniform_noise(shape=(2,),ranger=(0,1))
        self.noise=self.noise_generater.generate(self.number)
        
        interval=(np.max(ranger)-np.min(ranger))/number
        self.x=np.array([np.min(ranger)+i*interval for i in range(number)])
        
        self.y=np.array([self.func(i) for i in self.x]) 
        self.data=np.vstack((self.y,self.x)).T
        
        self.noise=self.noise_generater.generate(number)
        
class Single_poly_mock(Single_base_mock):
    def __init__(self,ranger:tuple, weight:list, number:int, noise_generator:Noise=None, *args, **kwargs) -> None:
        ''' Initialize single polynome x y mock data class
            data is: (number, 2) ,data[i] is [y,x]
        Args:
            // rank: x order, eg. rank=2 -> y=x^2+x+C where C is a constraint
            weight: the weight of polynome, if y=b+w1*x+w2*x^2, then w[0]=b,
            number: how many data you want to generate
            noise: noise generator
        '''        
        self.func=lambda x : np.sum([w*x**i for i,w in enumerate(weight)])
        print(type(self.func))
        super().__init__(ranger=ranger, number=number, noise_generator=noise_generator,func=self.func, args=args, kwargs=kwargs)
        # shape=(2,) (y,x)
 
class Single_sin_mock(Single_base_mock):
    def __init__(self,x_ranger:tuple, number:int,y_ranger:tuple=(-1,1), noise_generator:Noise=None, *args, **kwargs) -> None:
        self.mean=np.mean(y_ranger)
        self.scale=np.max(y_ranger)-np.min(y_ranger)
        
        self.func=lambda x : np.sin(x)
        
        super().__init__(ranger=x_ranger, number=number, noise_generator=noise_generator, func=self.func, args=args, kwargs=kwargs)

        self.y=self.y*self.scale+self.mean
        self.data=np.vstack((self.y,self.x)).T
    
if __name__=="__main__":
    # noiser=Gauss_noise((2,2),(-1,4))
    # print(noiser.generate())
    # mock=Single_poly_mock((0,10),[1,1,1,1],20)
    mock=Single_sin_mock((0,10),5)
    print(mock.generate())
    print(mock.generate_no_noise())