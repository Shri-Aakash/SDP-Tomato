#!/usr/bin/env python2
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import numpy as np

a1 = 19.75 #length of link a1 in cm
a2 = 19.85 #length of link a2 in cm

def q2(x,y):
	return np.arccos((x**2+y**2-a1**2-a2**2)/(2*a1*a2))

def q1(x,y,q2):
	if x==0:
		return 90-np.arctan(a2*np.sin(q2)/(a1+a2*np.cos(q2)))
	return (np.arctan(y/x)-np.arctan(a2*np.sin(q2)/(a1+a2*np.cos(q2))))

def convertToTicks(angle):
	return round(4096/360*angle)

if __name__=='__main__':
	try:
		x,y=0,0
		rospy.init_node('Ticks_Pub',anonymous=True)
		pub=rospy.Publisher('/Gesture_Info',Int32MultiArray,queue_size=10)
		data_to_send=Int32MultiArray()
		while not rospy.is_shutdown():
			try:
				x=float(input("Enter x coordinate"))
				y=float(input("Enter y coordinate"))
				print(x,y)
				ang2=q2(x,y)
				print(ang2)
				ang1=q1(x,y,ang2)
				print(ang1)
				t1=convertToTicks(ang1)
				t2=convertToTicks(ang2)
				print(t1,t2)
			except:
				print("Enter valid Coordinates")
		data_to_send.data=[t1,t2]
		pub.publish(data_to_send)
		rospy.sleep(2)
	except rospy.ROSInterruptException:
		rospy.loginfo('Execution interrupted. Try again')


