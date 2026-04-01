#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16


def main():
    rospy.init_node("respawn_objects_once")

    topic = rospy.get_param("~topic", "/rviz_panel/respawn_objects")
    cmd = rospy.get_param("~cmd", 1)   # 1=spawn, 0=delete
    delay = rospy.get_param("~delay", 1.0)

    pub = rospy.Publisher(topic, Int16, queue_size=1, latch=True)

    rospy.loginfo("Waiting %.1f seconds before publishing to %s ...", delay, topic)
    rospy.sleep(delay)

    msg = Int16()
    msg.data = int(cmd)

    pub.publish(msg)
    rospy.loginfo("Published %d to %s", msg.data, topic)

    rospy.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass