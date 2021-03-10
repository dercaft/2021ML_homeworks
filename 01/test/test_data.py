import os
import sys
import numpy as np
FATHERPATH=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FATHERPATH)
from model import base

if(not __name__=="__main__"): # 避免 Pylance 路径找不到
    from ..model import base

path=os.path.join(FATHERPATH,"data.txt")
lr=base.linear_regression(1)
lr.load_data(path=path)
print(lr.predict(5))