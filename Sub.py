#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
#sys.path.append('~/ajit_ws/src/dynamixel_workbench_msgs/msg')
#sys.path.append('~/ajit_ws/src/dynamixel_workbench_msgs/msg')
from dynamixel_workbench_msgs.msg import DynamixelStateList
from dynamixel_workbench_msgs.srv import DynamixelCommand,DynamixelCommandRequest
Ajit_Pose_client=rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command',DynamixelCommand)

gesture=0
def command(m_id,val):
	Ajit_Pose=DynamixelCommandRequest()
	Ajit_Pose.id=m_id
	Ajit_Pose.addr_name='Goal_Position'
	Ajit_Pose.value=val
	a=Ajit_Pose_client(Ajit_Pose)


def AjitCallback(info):
	for lst in info.lists:
		i=1
		for ele in lst.elements:
			command(i,ele)

if __name__=='__main__':
	try:
		while not rospy.is_shutdown():
			rospy.init_node('Gesture_sub',anonymous=True)
			rospy.Subscriber('/Gesture_Info',Int32MultiArray,AjitCallback)
			rospy.spin()
	except:
		pass

