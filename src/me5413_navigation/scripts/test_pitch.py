#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion


class TestPitch:
    def __init__(self):
        imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=1)
        rospy.loginfo("subscribe imu topic: %s", imu_topic)

    def imu_callback(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        rospy.loginfo_throttle(
            0.2,
            "roll=%.2f deg, pitch=%.2f deg, yaw=%.2f deg",
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        )


if __name__ == "__main__":
    rospy.init_node("test_pitch")
    TestPitch()
    rospy.spin()