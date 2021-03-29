import os
import sys
import numpy as np

DEBUG=1

class base:
    def __init__(self, *args, **kwargs) -> None:
        self.data=None
        self.preprocess=lambda x:x if not kwargs.__contains__("preprocess") else kwargs["preprocess"]            
        pass
    def fit(self,data):
        raise NotImplementedError
    def forward(self,x):
        raise NotImplementedError
    def update(self,loss):
        raise NotImplementedError
    def predict(self,x):
        raise NotImplementedError
    
class linear_regression(base):
    def __init__(self,rank:int=1, l2:float=0, *args, **kwargs) -> None:
        ''' w[0] x**0的系数（即b）w[1] x**1的系数 '''
        super().__init__(*args, **kwargs)
        assert rank>=0
        self._rank=rank
        self._w=np.zeros(rank+1)
        self._l2=l2*np.identity(rank+1).squeeze()
        self._l2_float=l2
        
    @property
    def l2(self):
        return self._l2_float
    @property
    def w(self):
        return self._w
    @property
    def rank(self):
        return self._rank
    
    def _vandermonde(self,x):
        ''' Calculate vandermonde linspace
        Args:
            x: (n,)
        Returns:
            r: (n,k)
        '''
        func=lambda k :[k**i for i in range(self._rank+1)]
        r=np.array([func(j) for j in x]).squeeze()
        return r
    
    def fit(self,data):
        ''' least square method 
        Args:
            data: (n,2), data[0]=(y,x)
        '''
        x=data[:,1]
        y=data[:,0]
        r=self._vandermonde(x)
        invertor=np.dot(r.T,r)
        if(len(invertor.shape)==0):
            w=(1/invertor+self._l2)*r.T@y
        else:
            w=np.linalg.inv(invertor+self._l2)@r.T@y
        self._w=w
        
    def predict(self,x):
        r=self._vandermonde(x)
        y=np.dot(r,self._w)
        return y
    
class Curve_fitter(base):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    