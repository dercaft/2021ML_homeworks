import os
import sys
import numpy as np

DEBUG=1

class base:
    def __init__(self, *args, **kwargs) -> None:
        self.data=None
        self.preprocess=lambda x:x if not kwargs.__contains__("preprocess") else kwargs["preprocess"]            
        pass
    def load_data(self, path:str, *args, **kwargs):
        temp=None
        with open(path,'rb') as f:
            temp=f.readlines()
        if(not len(temp)): 
            print("load nothing!") 
            return
        self.data=self.preprocess(temp)
        print("load {} data".format(len(self.data)))
        pass
class linear_regression(base):
    def __init__(self,rank:int=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w=np.zeros(rank)
        if(DEBUG):
            self.w=np.ones(rank)
        self.b=0
    def fit(self):
        pass
    def predict(self,x):
        r=[x**(i+1) for i in range(len(self.w))]
        y=np.dot(r,self.w)+self.b
        return y