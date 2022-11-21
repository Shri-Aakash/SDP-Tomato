#!/usr/bin/env python3
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from math import acos,atan,atan2,sin,cos
from math import radians,degrees

a1 = 0.1975 #length of link a1 in cm
a2 = 0.1985 #length of link a2 in cm

def q2(x,y):
    return acos((x**2+y**2-(a1**2+a2**2))/(2*a1*a2))

def q1(x,y,t2):
    gamma=atan2(y, x)
    beta=atan2(a2*sin(t2),a1+(2*a2*cos(t2)))
    return gamma-beta

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
				x,y=map(float,input("Enter coordinates: ").rstrip().split())
				print(x,y)
				theta2=q2(x,y)
				theta1=q1(x,y,theta1)
				print(theta1,theta2)
				t1=convertToTicks(theta1)
				t2=convertToTicks(theta2)
			except:
				print("Enter valid Coordinates")
		data_to_send.data=[t1,t2]
		pub.publish(data_to_send)
		rospy.sleep(2)
	except rospy.ROSInterruptException:
		rospy.loginfo('Execution interrupted. Try again')


