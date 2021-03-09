#!/usr/bin/env python
import numpy as np

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
