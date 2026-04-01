cmake_minimum_required(VERSION 3.0.2)
project(teleop_twist_keyboard)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/mbot_teleop.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
