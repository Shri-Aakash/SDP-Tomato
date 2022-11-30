# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:06:38 2022

@author: aakaa
"""
from math import acos,atan,atan2,sin,cos
from math import radians,degrees
#import numpy as np
#import os

a1=19.75
a2=19.85

max_reach=a1+a2
min_reach=abs(a1-a2)

def q2(x,y):
    return acos((x**2+y**2-(a1**2+a2**2))/(2*a1*a2))

def q1(x,y,t2):
    gamma=atan2(y, x)
    beta=atan2(a2*sin(t2),a1+(2*a2*cos(t2)))
    return gamma-beta

if __name__=='__main__':
    x,y=map(float,input().rstrip().split())
    try:
        theta2=q2(x,y)
        theta1=q1(x,y,theta2)
        print(degrees(theta1),degrees(theta2))
    except:
        print("No valid solution for given co-ordinates.Make sure that co-ordinates are correct")
    