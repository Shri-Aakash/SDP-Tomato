#!/usr/bin/env python
from math import acos,atan,atan2,sin,cos
from math import radians,degrees

a1 = 19.75 #length of link a1 in m
a2 = 19.85 #length of link a2 in m

def q2(x,y):
    return acos((x**2+y**2-(a1**2+a2**2))/(2*a1*a2))

def q1(x,y,t2):
    gamma=atan2(y, x)
    beta=atan2(a2*sin(t2),a1+(2*a2*cos(t2)))
    return gamma-beta

def convertToTicks(angle):
	return round((4096/360)*angle)

if __name__=='__main__':
	try:
		x,y=0,0
		while True:
			try:
				x=float(input("Enter X: "))
				y=float(input("Enter Y: "))
				print(x,y)
				theta2=round(degrees(q2(x,y)))
				theta1=round(degrees(q1(x,y,theta2)))
				print(theta1,theta2)
				tm1=convertToTicks(theta1)
				tm2=convertToTicks(theta2)+2048
			except:
				print("Enter valid Coordinates")
	except :
		print('Execution interrupted. Try again')


