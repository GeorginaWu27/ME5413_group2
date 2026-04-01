#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import actionlib
import tf

from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32

class PatrolWaypointsNode(object):
    def __init__(self):
        rospy.init_node("patrol_waypoints")
        self.slope_mode_topic = rospy.get_param("~slope_mode_topic", "/slope_mode_controller/mode")
        self.slope_mode_value = 0
        self.slope_mode_enabled = False
        rospy.Subscriber(self.slope_mode_topic, Int32, self.slope_mode_cb, queue_size=1)
        # ----------------------------
        # 参数
        # ----------------------------
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.costmap_topic = rospy.get_param("~costmap_topic", "/move_base/global_costmap/costmap")
        self.amcl_topic = rospy.get_param("~amcl_topic", "/amcl_pose")

        self.loop_patrol = rospy.get_param("~loop_patrol", True)
        self.retry_num = rospy.get_param("~retry_num", 2)
        self.control_rate = rospy.get_param("~control_rate", 5.0)

        self.default_timeout = rospy.get_param("~default_timeout", 15.0)
        self.default_tolerance = rospy.get_param("~default_tolerance", 0.7)
        self.default_yaw_tolerance = rospy.get_param("~default_yaw_tolerance", 3.14)
        self.default_required = rospy.get_param("~default_required", False)

        # 障碍物 0.8m x 0.8m，替代目标搜索半径从 0.8m 开始更合理
        self.search_radii = rospy.get_param("~search_radii", [0.8, 1.0, 1.2, 1.5])
        self.search_angle_step_deg = rospy.get_param("~search_angle_step_deg", 30.0)

        # costmap 代价值阈值
        # 0~free_cost_threshold 看成可用
        self.free_cost_threshold = rospy.get_param("~free_cost_threshold", 20)

        self.waypoints = rospy.get_param("~waypoints", [])
        if not self.waypoints:
            rospy.logerr("No waypoints in patrol params.")
            raise RuntimeError("No waypoints configured.")

        # ----------------------------
        # 运行时状态
        # ----------------------------
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.costmap = None
        self.costmap_info = None

        # 订阅
        rospy.Subscriber(self.amcl_topic, PoseWithCovarianceStamped, self.amcl_cb, queue_size=1)
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_cb, queue_size=1)

        # move_base action client
        rospy.loginfo("Waiting for move_base action server...")
        self.mb_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.mb_client.wait_for_server()
        rospy.loginfo("Connected to move_base.")

        rospy.loginfo("Waiting for AMCL pose and costmap...")
        self.wait_until_ready()
        rospy.loginfo("Patrol waypoints node ready.")

    # =========================================================
    # callbacks
    # =========================================================
    def amcl_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw

    def costmap_cb(self, msg):
        self.costmap = msg.data
        self.costmap_info = msg.info

    def wait_until_ready(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.robot_x is not None and self.costmap is not None:
                return
            rate.sleep()

    def slope_mode_cb(self, msg):
        self.slope_mode_value = msg.data
        self.slope_mode_enabled = (msg.data == 1)   # 1 == SLOPE_MODE

    # =========================================================
    # utils
    # =========================================================
    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def dist(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def yaw_diff(self, yaw1, yaw2):
        return abs(self.normalize_angle(yaw1 - yaw2))

    def yaw_to_quaternion(self, yaw):
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        quat = Quaternion()
        quat.x = q[0]
        quat.y = q[1]
        quat.z = q[2]
        quat.w = q[3]
        return quat

    def world_to_map(self, wx, wy):
        if self.costmap_info is None:
            return None

        origin_x = self.costmap_info.origin.position.x
        origin_y = self.costmap_info.origin.position.y
        resolution = self.costmap_info.resolution
        width = self.costmap_info.width
        height = self.costmap_info.height

        mx = int((wx - origin_x) / resolution)
        my = int((wy - origin_y) / resolution)

        if mx < 0 or my < 0 or mx >= width or my >= height:
            return None
        return mx, my

    def cost_at(self, wx, wy):
        idx = self.world_to_map(wx, wy)
        if idx is None:
            return 100

        mx, my = idx
        linear = my * self.costmap_info.width + mx

        if linear < 0 or linear >= len(self.costmap):
            return 100
        return self.costmap[linear]

    def is_pose_free(self, wx, wy):
        """
        -1 未知区域，这里保守处理成不可用
        """
        c = self.cost_at(wx, wy)
        if c < 0:
            return False
        return c <= self.free_cost_threshold

    def within_tolerance(self, goal_x, goal_y, goal_yaw, xy_tol, yaw_tol):
        if self.robot_x is None:
            return False

        d = self.dist(self.robot_x, self.robot_y, goal_x, goal_y)
        if d > xy_tol:
            return False

        if yaw_tol >= math.pi:
            return True

        if self.robot_yaw is None:
            return False

        return self.yaw_diff(self.robot_yaw, goal_yaw) <= yaw_tol

    def build_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation = self.yaw_to_quaternion(yaw)
        return goal

    # =========================================================
    # waypoint logic
    # =========================================================
    def find_alternative_goal(self, nominal_x, nominal_y):
        """
        原始点可用 -> 直接返回
        原始点被占 -> 在周围圆环找替代点
        """
        if self.is_pose_free(nominal_x, nominal_y):
            return nominal_x, nominal_y

        rospy.logwarn("Nominal point occupied: (%.2f, %.2f), searching nearby...", nominal_x, nominal_y)

        step = math.radians(self.search_angle_step_deg)

        for r in self.search_radii:
            angle = 0.0
            while angle < 2.0 * math.pi:
                x = nominal_x + r * math.cos(angle)
                y = nominal_y + r * math.sin(angle)

                if self.is_pose_free(x, y):
                    rospy.loginfo("Alternative goal found: (%.2f, %.2f), radius=%.2f", x, y, r)
                    return x, y

                angle += step

        rospy.logwarn("No alternative goal found around waypoint (%.2f, %.2f)", nominal_x, nominal_y)
        return None

    def send_goal_and_wait(self, nav_x, nav_y, nav_yaw,
                           nominal_x, nominal_y, nominal_yaw,
                           xy_tol, yaw_tol, timeout_sec):
        """
        nav_x/nav_y 是实际给 move_base 的目标
        nominal_x/nominal_y 是原始巡逻点
        到原始点附近 or 到替代点附近，都算成功
        """
        goal = self.build_goal(nav_x, nav_y, nav_yaw)
        self.mb_client.send_goal(goal)

        start = rospy.Time.now()
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            if self.slope_mode_enabled:
                rospy.logwarn("Slope mode detected, terminate current waypoint immediately.")
                self.mb_client.cancel_goal()
                return False, "slope_mode"

            elapsed = (rospy.Time.now() - start).to_sec()

            # 到原始参考点附近
            if self.within_tolerance(nominal_x, nominal_y, nominal_yaw, xy_tol, yaw_tol):
                rospy.loginfo("Reached nominal waypoint tolerance.")
                self.mb_client.cancel_goal()
                return True, "reached_nominal"

            # 到替代导航点附近
            if self.within_tolerance(nav_x, nav_y, nav_yaw, xy_tol, yaw_tol):
                rospy.loginfo("Reached alternative/nav goal tolerance.")
                self.mb_client.cancel_goal()
                return True, "reached_alternative"

            state = self.mb_client.get_state()

            # actionlib_msgs/GoalStatus:
            # 3 SUCCEEDED
            # 4 ABORTED
            # 5 REJECTED
            # 2 PREEMPTED
            if state == 3:
                return True, "move_base_succeeded"

            if state in [4, 5]:
                return False, "move_base_failed"

            if elapsed > timeout_sec:
                rospy.logwarn("Goal timeout: %.1f sec", elapsed)
                self.mb_client.cancel_goal()
                return False, "timeout"

            rate.sleep()

        return False, "shutdown"

    def parse_wp(self, wp):
        return {
            "name": str(wp.get("name", "wp")),
            "x": float(wp["x"]),
            "y": float(wp["y"]),
            "yaw": float(wp.get("yaw", 0.0)),
            "tolerance": float(wp.get("tolerance", self.default_tolerance)),
            "yaw_tolerance": float(wp.get("yaw_tolerance", self.default_yaw_tolerance)),
            "required": bool(wp.get("required", self.default_required)),
            "timeout": float(wp.get("timeout", self.default_timeout)),
        }

    def handle_one_waypoint(self, wp):
        name = wp["name"]
        nominal_x = wp["x"]
        nominal_y = wp["y"]
        nominal_yaw = wp["yaw"]
        xy_tol = wp["tolerance"]
        yaw_tol = wp["yaw_tolerance"]
        required = wp["required"]
        timeout_sec = wp["timeout"]

        rospy.loginfo("========== Waypoint [%s] (%.2f, %.2f) ==========", name, nominal_x, nominal_y)

        # 已经在附近，直接成功
        if self.within_tolerance(nominal_x, nominal_y, nominal_yaw, xy_tol, yaw_tol):
            rospy.loginfo("Already within tolerance of [%s]", name)
            return True

        for i in range(self.retry_num + 1):
            rospy.loginfo("[%s] attempt %d / %d", name, i + 1, self.retry_num + 1)

            alt = self.find_alternative_goal(nominal_x, nominal_y)
            if alt is None:
                if required:
                    rospy.logerr("[%s] required waypoint blocked, and no alternative found.", name)
                    return False
                else:
                    rospy.logwarn("[%s] optional waypoint skipped because blocked.", name)
                    return True

            nav_x, nav_y = alt
            nav_yaw = math.atan2(nominal_y - nav_y, nominal_x - nav_x)

            # 如果替代点刚好就是原点，避免 atan2(0,0) 方向无意义
            if abs(nav_x - nominal_x) < 1e-6 and abs(nav_y - nominal_y) < 1e-6:
                nav_yaw = nominal_yaw

            ok, reason = self.send_goal_and_wait(
                nav_x, nav_y, nav_yaw,
                nominal_x, nominal_y, nominal_yaw,
                xy_tol, yaw_tol, timeout_sec
            )

            if ok:
                rospy.loginfo("[%s] success: %s", name, reason)
                return True
            else:
                if reason == "slope_mode":
                    rospy.logwarn("[%s] terminated by slope mode.", name)
                    return False
                rospy.logwarn("[%s] failed this attempt: %s", name, reason)

        if required:
            rospy.logerr("[%s] failed finally, and it is required.", name)
            return False
        else:
            rospy.logwarn("[%s] failed finally, but it is optional, skip.", name)
            return True

    def spin(self):
        rate = rospy.Rate(1.0)

        while not rospy.is_shutdown():
            for raw_wp in self.waypoints:
                wp = self.parse_wp(raw_wp)
                ok = self.handle_one_waypoint(wp)

                if not ok:
                    if self.slope_mode_enabled:
                        rospy.logwarn("Patrol stopped because slope mode was entered: %s", wp["name"])
                    else:
                        rospy.logerr("Patrol stopped because required waypoint failed: %s", wp["name"])
                    return

                rospy.sleep(0.5)

            if not self.loop_patrol:
                rospy.loginfo("Patrol finished once.")
                return

            rospy.loginfo("One patrol loop finished, restart...")
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PatrolWaypointsNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass