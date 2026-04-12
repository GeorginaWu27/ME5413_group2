#!/usr/bin/env python3
import os
import sys

binary = "/home/ros/workspace/ros/ME5413_Final_Project/devel_isolated/cartographer_ros/lib/cartographer_ros/cartographer_occupancy_grid_node"
os.execv(binary, [binary] + sys.argv[1:])

