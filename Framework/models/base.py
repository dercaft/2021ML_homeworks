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
        super().__init__(*args, **kwargs)
        assert rank>=0
        self.rank=rank
        self.w=np.zeros(rank+1)
        self.l2=l2*np.identity(rank+1).squeeze()
        
    def _vandermonde(self,x):
        ''' Calculate vandermonde linspace
        Args:
            x: (n,)
        Returns:
            r: (n,k)
        '''
        func=lambda k :[k**i for i in range(self.rank+1)]
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
            w=(1/invertor+self.l2)*r.T@y
        else:
            w=np.linalg.inv(invertor+self.l2)@r.T@y
        self.w=w
        
    def predict(self,x):
        r=self._vandermonde(x)
        y=np.dot(r,self.w)
        return y