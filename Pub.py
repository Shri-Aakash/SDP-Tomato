#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from math import acos,atan,atan2,sin,cos
from math import radians,degrees

a1 = 0.1975 #length of link a1 in m
a2 = 0.1985 #length of link a2 in m

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
		rospy.init_node('Ticks_Pub',anonymous=True)
		pub=rospy.Publisher('/Scara_angles',Int32MultiArray,queue_size=10)
		data_to_send=Int32MultiArray()
		while not rospy.is_shutdown():
			try:
				x=float(input("Enter X: "))
				y=float(input("Enter Y: "))
				print(x,y)
				theta2=round(degrees(q2(x,y)))
				theta1=round(degrees(q1(x,y,theta2)))
				print(theta1,theta2)
				tm1=convertToTicks(theta1)
				tm2=convertToTicks(theta2)+2048
				data_to_send.data=[tm1,tm2]
				pub.publish(data_to_send)
			except:
				print("Enter valid Coordinates")
		rospy.sleep(2)
	except rospy.ROSInterruptException:
		rospy.loginfo('Execution interrupted. Try again')


