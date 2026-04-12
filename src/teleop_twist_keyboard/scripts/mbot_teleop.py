#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module implements keyboard teleoperation for the mbot robot.
It captures keyboard inputs and publishes Twist messages to the '/cmd_vel' topic.
"""

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

# Instruction manual displayed in the terminal
msg = """
Control mbot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

# Mapping of keys to movement directions (linear, angular)
moveBindings = {
        'i':(1,0),
        'o':(1,-1),
        'j':(0,1),
        'l':(0,-1),
        'u':(1,1),
        ',':(-1,0),
        '.':(-1,1),
        'm':(-1,-1),
           }

# Mapping of keys to speed scale factors (linear, angular)
speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
          }

def getKey():
    """
    Reads a single key stroke from the terminal without requiring Enter.
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

speed = .2
turn = 1

def vels(speed,turn):
    """
    Returns a formatted string of the current speed settings.
    """
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    
    # Initialize ROS node and publisher
    rospy.init_node('mbot_teleop')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    x = 0
    th = 0
    status = 0
    count = 0
    acc = 0.1
    target_speed = 0
    target_turn = 0
    control_speed = 0
    control_turn = 0
    try:
        print (msg)
        print (vels(speed,turn))
        while(1):
            key = getKey()
            # Movement control keys (1: positive direction, -1: negative direction)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0
            # Speed modification keys
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]  # Increase/decrease linear speed
                turn = turn * speedBindings[key][1]    # Increase/decrease angular speed
                count = 0

                print (vels(speed,turn))
                if (status == 14):
                    print (msg)
                status = (status + 1) % 15
            # Stop keys
            elif key == ' ' or key == 'k' :
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0
            else:
                count = count + 1
                if count > 4:
                    x = 0
                    th = 0
                if (key == '\x03'):
                    break

            # Calculate target velocities: speed magnitude * direction
            target_speed = speed * x
            target_turn = turn * th

            # Speed ramping logic to prevent sudden jerky movements
            if target_speed > control_speed:
                control_speed = min( target_speed, control_speed + 0.02 )
            elif target_speed < control_speed:
                control_speed = max( target_speed, control_speed - 0.02 )
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min( target_turn, control_turn + 0.1 )
            elif target_turn < control_turn:
                control_turn = max( target_turn, control_turn - 0.1 )
            else:
                control_turn = target_turn

            # Create and publish the Twist message
            twist = Twist()
            twist.linear.x = control_speed; 
            twist.linear.y = 0; 
            twist.linear.z = 0
            twist.angular.x = 0; 
            twist.angular.y = 0; 
            twist.angular.z = control_turn
            pub.publish(twist)

    except:
        print (e)

    finally:
        # Ensure the robot stops when the script exits
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)

    # Restore terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)