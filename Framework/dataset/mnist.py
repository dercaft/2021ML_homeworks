# -*-coding:utf-8 -*-

import os
import sys
import struct
import numpy as np

TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL= "train-labels.idx1-ubyte"
VALID_DATA = "t10k-images.idx3-ubyte"
VALID_LABEL= "t10k-labels.idx1-ubyte"


class data_base:
    def __getitem__(self,index):
        raise NotImplementedError
    
    # def __iter__(self):
    #     raise NotImplementedError

class MNIST(data_base):
    ''' MNIST dataset class
        load all data into memory when initialed
    '''
    def __init__(self,path:str,typer:str="train") -> None:
        ''' initialization
        Args:
            path: path of dataset directory
        '''
        super().__init__()
        
        self.t_data, self.t_len, self.h , self.w    = self._image_parser(path,TRAIN_DATA)
        self.v_data, self.v_len, self.v_h, self.v_w = self._image_parser(path,VALID_DATA)
        
        self.t_label, self.t_l_len = self._label_parser(path,TRAIN_LABEL)
        self.v_label, self.v_l_len = self._label_parser(path,VALID_LABEL)
        
        if typer=="train":
            self.data_name=TRAIN_DATA
            self.label_name=TRAIN_LABEL
        elif typer=="valid":
            self.data_name=VALID_DATA
            self.label_name=VALID_LABEL
        else:
            raise AttributeError
        self.data, self.length, self.h, self.w = self._image_parser(path,self.data_name)
        self.label, self.l_length = self._label_parser(path,self.label_name)
        
        assert(self.length==self.l_length)

    def _get_image(self,buff,index:int,row:int,col:int):
        ''' get image from bit string
        Args:
            buff: cached bit string 
            index: denote where image start
            row: row size of image
            col: col size of image
        Returns:
            tuple, contains parsered image and index of next image
            img: np.array(dtype=np.uint8)
            index: int 
        '''
        bits='>'+'B'*(row*col)
        length=struct.calcsize(bits)
        index=index+length
        i=struct.unpack_from(bits,buff,index)
        img=np.array(i,dtype=np.uint8)
        img.resize((row,col))
        return (img,index)
    
    def _image_parser(self,path:str,file:str):
        ''' get all images from data file
        Args:
            path: dataset directory
            file: datafile name
        Returns:
            image_set: loaded images
            images: total image number
            rows: image row number
            columns: image column number
        '''
        file_path=os.path.join(path,file)
        cache,index=None,0
        with open(file_path,'rb') as f:
            cache=f.read()
        _,images,rows,columns=struct.unpack_from('>IIII',cache,index)
        index+=struct.calcsize('>IIII')
        
        image_set=[]
        for i in range(images):
            img,index=self._get_image(cache,index,rows,columns)
            image_set.append(img)
        return image_set,images,rows,columns
    
    def _label_parser(self,path:str,file:str):
        ''' get all labels from label file 
        Args are same as above
        Returns:
            label_set: label set
            labels: number of labels
        '''
        file_path=os.path.join(path,file)
        cache,index=None,0
        with open(file_path,'rb') as f:
            cache=f.read()
        _,labels = struct.unpack_from('>II',cache,index)
        index+=struct.calcsize('>II')
        
        label_set=[]
        for i in range(labels):
            l=int( struct.unpack_from('>B',cache,index) )
            index+=struct.calcsize('>B')
            label_set.append(l)
        return label_set,labels

    def __getitem__(self, index:int):
        return (self.data[index],self.label[index])
    
    def __len__(self):
        return self.length
