# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:03:37 2022

@author: aakaa
"""

import numpy as np
R180_X=[[1,0,0],[0,np.cos(np.pi),-np.sin(np.pi)],[0,np.sin(np.pi),np.cos(np.pi)]]
pi_by_2=(np.pi)/2
RZ_n90=[[np.cos(pi_by_2),-np.sin(pi_by_2),0],[np.sin(pi_by_2),np.cos(pi_by_2),0],[0,0,1]]
RO_C=np.dot(R180_X,RZ_n90)
dO_C=[[17.5],[20.5],[0]]

HO_C=np.concatenate((RO_C,dO_C),1)
HO_C=np.concatenate((HO_C,[[0,0,0,1]]),0)
def transform(x,y):
    PC=[[x],[y],[0],[1]]
    P0=np.dot(HO_C,PC)
    return P0[0],P0[1]


# print(transform(1,2))
