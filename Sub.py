#! /usr/bin/env python
import rospy
import sys
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
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


def ScaraCB(info):
	for ele in info.data:
		print(ele)
		#command(i,ele)

if __name__=='__main__':
	try:
		while not rospy.is_shutdown():
			rospy.init_node('Scara_angle_sub',anonymous=True)
			rospy.Subscriber('/Scara_angles',Int32MultiArray,ScaraCB)
			rospy.spin()
	except:
		pass

