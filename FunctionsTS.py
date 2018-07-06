# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:00:19 2018

@author: a688291
"""

import numpy as np
import pandas as pd


x_0 = np.zeros(9)
x_0.append(2)
x_0 = np.append(x_0,2)

def ForwardSeries(ts_base, ts_diff,diff_degree):
    len_ts  = len(ts_base)
    ts_base = ts_base[:diff_degree]
    new_ts = ts_base
    for i in range(diff_degree):
        new_ts[i] = new_ts[i] + ts_diff[i]
    for i in range(len_ts)-diff_degree:
        new_ts = np.append(new_ts,new_ts[i]+ts_diff[i+diff_degree])
        

        