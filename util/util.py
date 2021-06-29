#!/usr/bin/env python
import numpy as np
import os

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # select only the values that are greater than zero
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1=0.0
    if a2 != a2:
        a2=0.0
    if a3 != a3:
        a3=0.0
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel:
        abs_rel=0.0
    if log_10 != log_10:
        log_10=0.0
    if mae != mae:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae

class TextWrite(object):
    ''' Wrting the values to a text file 
    '''
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, "w+")
        self.file.close()
        self.str_write = ''
    
    def add_line_csv(self, data_list):
        str_tmp = []
        for item in data_list:
            if isinstance(item, int):
                str_tmp.append("{:03d}".format(item))
            if isinstance(item, str):
                str_tmp.append(item)
            if isinstance(item, float):
                str_tmp.append("{:.6f}".format(item))
        
        self.str_write = ",".join(str_tmp) + "\n"
    
    def add_line_txt(self, content, size=None, maxLength = 10, heading=False):
        if size == None:
            size = [1 for i in range(len(content))]
        if heading:    
            str_tmp = '|'.join(list(map(lambda x,s:x.center((s*maxLength)+(s-1)), content, size)))
        else:
            str_tmp = '|'.join(list(map(lambda x,s:x.rjust((s*maxLength)+(s-1)), content, size)))
        self.str_write += str_tmp + "\n" 

    def write_line(self):  
        self.file = open(self.filename, "a")
        self.file.write(self.str_write)
        self.file.close()
        self.str_write = ''

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)