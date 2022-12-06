#! /usr/bin/env python
from Inverse import *
import rospy
import pandas as pd
import sys
from math import degrees
import time
#sys.path.append('/home/aakash/dynamixel_ws/src/dynamixel-workbench-msgs/dynamixel_workbench_msgs')
from dynamixel_workbench_msgs.msg import DynamixelStateList
from dynamixel_workbench_msgs.srv import DynamixelCommand,DynamixelCommandRequest
Scara_Pose_client=rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command',DynamixelCommand)

def command(m_id,val):
	Scara_Pose=DynamixelCommandRequest()
	Scara_Pose.id=m_id
	Scara_Pose.addr_name='Goal_Position'
	Scara_Pose.value=val
	a=Scara_Pose_client(Scara_Pose)


def convertToTicks(ang):
	return round(4096/360*ang)

		

if __name__=='__main__':
	try:
		#rospy.init_node('Scara_angle_sub',anonymous=True)
		#rospy.Subscriber('/Scara_angles',Int32MultiArray,ScaraCB)
		#rospy.spin()
		df=pd.read_csv('Data.csv')
		for i in range(df.shape[0]):
			x=df.at[i,'X']
			y=df.at[i,'Y']
			cat=df.at[i,'Category']
			try:
				theta2=q2(x,y)
				theta1=q1(x,y,theta2)
				theta2=round(degrees(theta2))
				theta1=round(degrees(theta1))
				print(theta1,theta2)
				t1=convertToTicks(theta1)
				t2=convertToTicks(theta2)+2048
				
				if t1>=0 & t2>=0:
					command(1,t1)
					command(2,t2)
					time.sleep(3)
				else:
					print("Invalid Tick values.Please check the angles again")
				
			except:
				theta1=None
				theta2=None
				print(f'{x},{y} are not valid co-ordinates')
	except:
		pass


