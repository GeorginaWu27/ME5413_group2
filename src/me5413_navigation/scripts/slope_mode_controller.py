#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Int32
from std_srvs.srv import Empty
from dynamic_reconfigure.client import Client as DynClient

import math
import rospy
import actionlib
import tf

from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction
from tf.transformations import euler_from_quaternion


class SlopeModeController:
    NORMAL = 0
    SLOPE_MODE = 1
    RELOCALIZE = 2

    def __init__(self):
        # ---------- topic 参数 ----------
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.initialpose_topic = rospy.get_param("~initialpose_topic", "/initialpose")
        
        # ---------- 坡道判定参数 ----------
        self.slope_enter_deg = rospy.get_param("~slope_enter_deg", 6.0)
        self.slope_exit_deg = rospy.get_param("~slope_exit_deg", 3.0)
        self.enter_hold_sec = rospy.get_param("~enter_hold_sec", 0.5)
        self.exit_hold_sec = rospy.get_param("~exit_hold_sec", 1.0)

        # ---------- 坡道运动参数 ----------
        self.forward_speed = rospy.get_param("~forward_speed", 0.30)
        self.slow_speed = rospy.get_param("~slow_speed", 0.18)
        self.turn_speed = rospy.get_param("~turn_speed", 0.25)
        self.avoid_turn_speed = rospy.get_param("~avoid_turn_speed", 0.12)

        # ---------- 避障参数 ----------
        self.front_block_dist = rospy.get_param("~front_block_dist", 0.65)
        self.side_warn_dist = rospy.get_param("~side_warn_dist", 0.45)
        self.front_slow_dist = rospy.get_param("~front_slow_dist", 0.90)

        self.front_angle_deg = rospy.get_param("~front_angle_deg", 18.0)
        self.front_left_angle_deg = rospy.get_param("~front_left_angle_deg", 60.0)
        self.side_angle_deg = rospy.get_param("~side_angle_deg", 55.0)
        self.right_max_dist = rospy.get_param("~right_max_dist", 3.0)

        # ---------- 左墙跟随参数 ----------
        self.left_target_dist = rospy.get_param("~left_target_dist", 0.55)
        self.left_wall_kp = rospy.get_param("~left_wall_kp", 1.20)
        self.left_wall_kd = rospy.get_param("~left_wall_kd", 0.25)

        self.left_min_angle_deg = rospy.get_param("~left_min_angle_deg", 70.0)
        self.left_max_angle_deg = rospy.get_param("~left_max_angle_deg", 110.0)

        self.left_reacquire_dist = rospy.get_param("~left_reacquire_dist", 1.20)
        self.left_max_follow_dist = rospy.get_param("~left_max_follow_dist", 1.60)

        # ---------- 右绕与航向修正 ----------
        self.right_bypass_bias = rospy.get_param("~right_bypass_bias", 0.45)
        self.yaw_kp = rospy.get_param("~yaw_kp", 0.80)

        # ---------- 固定坡道方向 ----------
        self.slope_target_yaw_deg = rospy.get_param("~slope_target_yaw_deg", 0.0)
        self.slope_target_yaw_rad = math.radians(self.slope_target_yaw_deg)

        # ---------- 出坡重定位参数 ----------
        self.rotate_speed = rospy.get_param("~rotate_speed", 0.30)
        self.rotate_total_deg = rospy.get_param("~rotate_total_deg", 180.0)

        # ---------- 控制频率 ----------
        self.control_rate = rospy.get_param("~control_rate", 15.0)

        # ---------- 固定 initialpose 参数 ----------
        self.use_fixed_initialpose = rospy.get_param("~use_fixed_initialpose", True)
        self.fixed_pose_x = rospy.get_param("~fixed_pose_x", 0.0)
        self.fixed_pose_y = rospy.get_param("~fixed_pose_y", 0.0)
        self.fixed_pose_yaw_deg = rospy.get_param("~fixed_pose_yaw_deg", 0.0)

        self.fixed_cov_x = rospy.get_param("~fixed_cov_x", 0.8)
        self.fixed_cov_y = rospy.get_param("~fixed_cov_y", 0.8)
        self.fixed_cov_yaw_deg = rospy.get_param("~fixed_cov_yaw_deg", 25.0)

        self.publish_initialpose_on_flat = rospy.get_param("~publish_initialpose_on_flat", True)
        
        # ---------- 回到 NORMAL 后自动发送导航目标 ----------
        self.auto_send_goal_on_normal = rospy.get_param("~auto_send_goal_on_normal", True)

        self.normal_goal_x = rospy.get_param("~normal_goal_x", 32.10352325439453)
        self.normal_goal_y = rospy.get_param("~normal_goal_y", 6.640366077423096)
        self.normal_goal_z = rospy.get_param("~normal_goal_z", 0.0)

        self.normal_goal_qx = rospy.get_param("~normal_goal_qx", 0.0)
        self.normal_goal_qy = rospy.get_param("~normal_goal_qy", 0.0)
        self.normal_goal_qz = rospy.get_param("~normal_goal_qz", -0.9974967402639787)
        self.normal_goal_qw = rospy.get_param("~normal_goal_qw", 0.070712468226874)
        
        # ---------- 状态 ----------
        self.mode = self.NORMAL
        self.mode_pub = rospy.Publisher("~mode", Int32, queue_size=1, latch=True)
        self.mode_pub.publish(Int32(self.mode))
        
        self.has_imu = False
        self.has_scan = False

        self.roll_rad = 0.0
        self.pitch_rad = 0.0
        self.yaw_rad = 0.0
        self.latest_scan = None

        self.slope_enter_time = None
        self.flat_enter_time = None

        self.rotate_accum = 0.0
        self.last_yaw = None

        self.initialpose_sent_this_cycle = False
        self.last_left_error = 0.0

        # ---------- 通信 ----------
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.initialpose_pub = rospy.Publisher(self.initialpose_topic, PoseWithCovarianceStamped, queue_size=1)

        # rospy.loginfo("imu_topic: %s", self.imu_topic)
        # rospy.loginfo("scan_topic: %s", self.scan_topic)
        # rospy.loginfo("cmd_vel_topic: %s", self.cmd_vel_topic)
        # rospy.loginfo("initialpose_topic: %s", self.initialpose_topic)
        # rospy.loginfo("slope_target_yaw_deg: %.2f", self.slope_target_yaw_deg)

        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")

        # ---------- 楼层/地图参数切换 ----------
        self.current_floor = 1
        self.global_map_config_applied = 1

        # 一楼保持原参数
        self.floor1_footprint_padding = rospy.get_param("~floor1_footprint_padding", 0.10)
        self.floor1_inflation_radius = rospy.get_param("~floor1_inflation_radius", 0.30)
        self.floor1_cost_scaling_factor = rospy.get_param("~floor1_cost_scaling_factor", 10.0)

        # 二楼参数
        self.floor2_footprint_padding = rospy.get_param("~floor2_footprint_padding", 0.18)
        self.floor2_inflation_radius = rospy.get_param("~floor2_inflation_radius", 0.55)
        self.floor2_cost_scaling_factor = rospy.get_param("~floor2_cost_scaling_factor", 6.0)

        self.move_base_ns = rospy.get_param("~move_base_ns", "/move_base")

    # =========================
    # 回调
    # =========================
    def imu_callback(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        self.roll_rad = roll
        self.pitch_rad = pitch
        self.yaw_rad = yaw
        self.has_imu = True

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.has_scan = True

    # =========================
    # 工具函数
    # =========================
    @staticmethod
    def normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def finite_range(r):
        return (not math.isinf(r)) and (not math.isnan(r)) and r > 0.02

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def publish_cmd(self, vx, wz):
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def cancel_navigation(self):
        try:
            self.move_base_client.cancel_all_goals()
            rospy.logwarn("Slope mode: cancel all move_base goals.")
        except Exception as e:
            rospy.logwarn("Failed to cancel move_base goals: %s", str(e))

    def get_min_range_deg(self, angle_min_deg, angle_max_deg):
        if not self.has_scan or self.latest_scan is None:
            return float("inf")

        scan = self.latest_scan
        a0 = math.radians(min(angle_min_deg, angle_max_deg))
        a1 = math.radians(max(angle_min_deg, angle_max_deg))

        start_idx = int((a0 - scan.angle_min) / scan.angle_increment)
        end_idx = int((a1 - scan.angle_min) / scan.angle_increment)

        start_idx = max(0, start_idx)
        end_idx = min(len(scan.ranges) - 1, end_idx)

        if end_idx < start_idx:
            return float("inf")

        min_r = float("inf")
        for i in range(start_idx, end_idx + 1):
            r = scan.ranges[i]
            if self.finite_range(r) and r < min_r:
                min_r = r
        return min_r

    def get_median_range_deg(self, angle_min_deg, angle_max_deg):
        if not self.has_scan or self.latest_scan is None:
            return float("inf")

        scan = self.latest_scan
        a0 = math.radians(min(angle_min_deg, angle_max_deg))
        a1 = math.radians(max(angle_min_deg, angle_max_deg))

        start_idx = int((a0 - scan.angle_min) / scan.angle_increment)
        end_idx = int((a1 - scan.angle_min) / scan.angle_increment)

        start_idx = max(0, start_idx)
        end_idx = min(len(scan.ranges) - 1, end_idx)

        if end_idx < start_idx:
            return float("inf")

        vals = []
        for i in range(start_idx, end_idx + 1):
            r = scan.ranges[i]
            if self.finite_range(r):
                vals.append(r)

        if not vals:
            return float("inf")

        vals.sort()
        return vals[len(vals) // 2]

    def get_tilt_deg(self):
        roll_deg = math.degrees(self.roll_rad)
        pitch_deg = math.degrees(self.pitch_rad)
        tilt_deg = math.sqrt(roll_deg * roll_deg + pitch_deg * pitch_deg)
        return tilt_deg

    def publish_initialpose(self, x, y, yaw_deg, cov_x, cov_y, cov_yaw_deg):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        yaw = math.radians(yaw_deg)
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.pose.covariance[0] = cov_x * cov_x
        msg.pose.covariance[7] = cov_y * cov_y
        msg.pose.covariance[35] = math.radians(cov_yaw_deg) ** 2

        self.initialpose_pub.publish(msg)

        rospy.logwarn(
            "Published /initialpose: x=%.2f y=%.2f yaw=%.1f deg cov=(%.2f, %.2f, %.1fdeg)",
            x, y, yaw_deg, cov_x, cov_y, cov_yaw_deg
        )

    # =========================
    # 模式判定
    # =========================
    def slope_condition_met(self):
        tilt_deg = self.get_tilt_deg()
        now = rospy.Time.now().to_sec()

        if tilt_deg > self.slope_enter_deg:
            if self.slope_enter_time is None:
                self.slope_enter_time = now
            if now - self.slope_enter_time >= self.enter_hold_sec:
                return True
        else:
            self.slope_enter_time = None

        return False

    def flat_condition_met(self):
        tilt_deg = self.get_tilt_deg()
        now = rospy.Time.now().to_sec()

        if tilt_deg < self.slope_exit_deg:
            if self.flat_enter_time is None:
                self.flat_enter_time = now
            if now - self.flat_enter_time >= self.exit_hold_sec:
                return True
        else:
            self.flat_enter_time = None

        return False

    # =========================
    # 模式切换
    # =========================
    def enter_slope_mode(self):
        self.cancel_navigation()
        self.stop_robot()
        self.mode = self.SLOPE_MODE
        self.flat_enter_time = None
        self.initialpose_sent_this_cycle = False
        self.last_left_error = 0.0
        self.mode_pub.publish(Int32(self.mode))
        rospy.logwarn("Enter SLOPE_MODE.")

    def enter_relocalize(self):
        self.stop_robot()

        if self.publish_initialpose_on_flat and self.use_fixed_initialpose and (not self.initialpose_sent_this_cycle):
            rospy.sleep(0.2)
            self.publish_initialpose(
                self.fixed_pose_x,
                self.fixed_pose_y,
                self.fixed_pose_yaw_deg,
                self.fixed_cov_x,
                self.fixed_cov_y,
                self.fixed_cov_yaw_deg
            )
            self.initialpose_sent_this_cycle = True
            rospy.sleep(0.3)

        self.mode = self.RELOCALIZE
        self.rotate_accum = 0.0
        self.last_yaw = None
        self.mode_pub.publish(Int32(self.mode))
        rospy.logwarn("Enter RELOCALIZE mode.")

    def exit_to_normal(self):
        self.stop_robot()
        self.mode = self.NORMAL
        self.slope_enter_time = None
        self.flat_enter_time = None
        self.rotate_accum = 0.0
        self.last_yaw = None
        self.initialpose_sent_this_cycle = False
        self.last_left_error = 0.0
        self.mode_pub.publish(Int32(self.mode))
        rospy.logwarn("Back to NORMAL mode.")

        # 这里认为：出坡完成后已经到二楼
        self.current_floor = 2
        # 先切 global_costmap 参数
        self.apply_global_costmap_params(self.current_floor)

        rospy.sleep(0.5)   # 可选，给 AMCL / move_base 一点恢复时间
        try:
            rospy.wait_for_service('/move_base/clear_costmaps')
            self.clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            self.clear_costmaps_srv()
            rospy.logwarn("Cleared move_base costmaps after relocalization.")
        except Exception as e:
            rospy.logwarn("Failed to clear costmaps: %s", str(e))
            
        rospy.sleep(0.3)
        if self.auto_send_goal_on_normal:
            self.send_normal_goal()
    
    def apply_global_costmap_params(self, floor_id):
        """
        只动态调整 global_costmap：
        1) footprint_padding
        2) inflation_radius
        3) cost_scaling_factor
        """
        try:
            if floor_id == 2:
                footprint_padding = self.floor2_footprint_padding
                inflation_radius = self.floor2_inflation_radius
                cost_scaling_factor = self.floor2_cost_scaling_factor
            else:
                footprint_padding = self.floor1_footprint_padding
                inflation_radius = self.floor1_inflation_radius
                cost_scaling_factor = self.floor1_cost_scaling_factor

            # 1. global_costmap 本体参数
            global_costmap_client = DynClient(
                self.move_base_ns + "/global_costmap",
                timeout=2.0
            )
            global_costmap_client.update_configuration({
                "footprint_padding": footprint_padding
            })

            # 2. inflation layer 参数
            inflation_client = DynClient(
                self.move_base_ns + "/global_costmap/inflater_layer",
                timeout=2.0
            )
            inflation_client.update_configuration({
                "inflation_radius": inflation_radius,
                "cost_scaling_factor": cost_scaling_factor
            })

            self.global_map_config_applied = floor_id

            rospy.logwarn(
                "Applied global_costmap params for floor %d: footprint_padding=%.3f, inflation_radius=%.3f, cost_scaling_factor=%.3f",
                floor_id, footprint_padding, inflation_radius, cost_scaling_factor
            )

        except Exception as e:
            rospy.logwarn("Failed to apply global_costmap params for floor %d: %s", floor_id, str(e))
    
    # =========================
    # 坡道模式控制
    # =========================
    def run_slope_mode(self):
        fa = self.front_angle_deg
        fla = self.front_left_angle_deg

        front_min = self.get_min_range_deg(-fa, fa)
        front_left_min = self.get_min_range_deg(0.0, fla)
        left_dist = self.get_median_range_deg(self.left_min_angle_deg, self.left_max_angle_deg)
        right_min = self.get_min_range_deg(-self.side_angle_deg, -fa)

        if math.isinf(right_min) or math.isnan(right_min) or right_min > self.right_max_dist:
            right_min = self.right_max_dist

        if math.isinf(left_dist) or math.isnan(left_dist):
            left_dist = self.left_max_follow_dist + 0.5

        yaw_error = self.normalize_angle(self.slope_target_yaw_rad - self.yaw_rad)

        # 前方被挡，或者左前有伸出的壁障
        front_blocked = (front_min < self.front_block_dist) or (front_left_min < self.side_warn_dist)
        front_slow = front_min < self.front_slow_dist
        left_wall_seen = left_dist < self.left_reacquire_dist

        vx = 0.0
        wz = 0.0

        if front_blocked:
            # 这种地图默认右绕，避免按左右对称选边后漂向右边大空场
            vx = self.slow_speed

            # 障碍越近，右转越强
            # dist_ratio = self.clamp((self.front_block_dist - front_min) / self.front_block_dist, 0.0, 1.0)
            # front_term = 0.25 + dist_ratio
            # wz = -front_term
            wz =-self.turn_speed

            # 如果已经离左墙太远，则减小右转，慢慢把车拉回左边
            if left_dist > self.left_max_follow_dist:
                # recover = self.clamp((left_dist - self.left_max_follow_dist) * 0.8, 0.0, self.turn_speed)
                # wz += recover
                vx = self.slow_speed
                wz =self.turn_speed
        else:
            vx = self.slow_speed if front_slow else self.forward_speed

            # 正常时贴左墙走
            left_error = left_dist - self.left_target_dist
            d_error = (left_error - self.last_left_error) * self.control_rate
            self.last_left_error = left_error

            wall_term = self.left_wall_kp * left_error + self.left_wall_kd * d_error
            wall_term = self.clamp(wall_term, -self.turn_speed, self.turn_speed)

            # left_error > 0 表示离左墙太远，需要左转，ROS中 wz>0 通常表示左转
            wz = wall_term

            # 再叠加一点朝坡道目标方向的修正，防止蛇形
            wz += self.clamp(self.yaw_kp * yaw_error, -0.25, 0.25)

            # 左墙暂时丢失，主动向左找墙
            if not left_wall_seen:
                wz += 0.18
                
            if left_dist > self.left_max_follow_dist:
                # recover = self.clamp((left_dist - self.left_max_follow_dist) * 0.8, 0.0, self.turn_speed)
                # wz += recover
                vx = self.slow_speed
                wz =self.turn_speed
        wz = self.clamp(wz, -self.turn_speed, self.turn_speed)
        vx = max(0.0, vx)

        self.publish_cmd(vx, wz)

        # rospy.loginfo_throttle(
        #     0.5,
        #     "mode=SLOPE tilt=%.2f front=%.2f front_left=%.2f left=%.2f right=%.2f vx=%.2f wz=%.2f yaw=%.2f target=%.2f",
        #     self.get_tilt_deg(),
        #     front_min, front_left_min, left_dist, right_min,
        #     vx, wz,
        #     math.degrees(self.yaw_rad),
        #     self.slope_target_yaw_deg
        # )

        if self.flat_condition_met():
            rospy.logwarn("Flat ground detected, switching to relocalization.")
            self.enter_relocalize()

    # =========================
    # 出坡后原地转圈重定位
    # =========================
    def run_relocalize(self):
        if not self.has_imu:
            self.stop_robot()
            return

        if self.last_yaw is None:
            self.last_yaw = self.yaw_rad

        delta = self.normalize_angle(self.yaw_rad - self.last_yaw)
        self.rotate_accum += abs(delta)
        self.last_yaw = self.yaw_rad

        target_rad = math.radians(self.rotate_total_deg)

        if self.rotate_accum >= target_rad:
            rospy.logwarn("Relocalization rotation finished.")
            self.stop_robot()
            rospy.sleep(1.0)
            self.exit_to_normal()
            return

        self.publish_cmd(0.0, self.rotate_speed)

    def send_normal_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"

        goal.target_pose.pose.position.x = self.normal_goal_x
        goal.target_pose.pose.position.y = self.normal_goal_y
        goal.target_pose.pose.position.z = self.normal_goal_z

        goal.target_pose.pose.orientation.x = self.normal_goal_qx
        goal.target_pose.pose.orientation.y = self.normal_goal_qy
        goal.target_pose.pose.orientation.z = self.normal_goal_qz
        goal.target_pose.pose.orientation.w = self.normal_goal_qw

        self.move_base_client.send_goal(goal)

        rospy.logwarn(
            "Sent goal after back to NORMAL mode: x=%.3f y=%.3f z=%.3f q=(%.3f, %.3f, %.3f, %.3f)",
            self.normal_goal_x,
            self.normal_goal_y,
            self.normal_goal_z,
            self.normal_goal_qx,
            self.normal_goal_qy,
            self.normal_goal_qz,
            self.normal_goal_qw
        )
    # =========================
    # 主循环
    # =========================
    def spin(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            if not self.has_imu or not self.has_scan:
                rate.sleep()
                continue

            if self.mode == self.NORMAL:
                # rospy.loginfo_throttle(
                #     1.0,
                #     "mode=NORMAL tilt=%.2f roll=%.2f pitch=%.2f yaw=%.2f",
                #     self.get_tilt_deg(),
                #     math.degrees(self.roll_rad),
                #     math.degrees(self.pitch_rad),
                #     math.degrees(self.yaw_rad)
                # )
                if self.slope_condition_met():
                    self.enter_slope_mode()

            elif self.mode == self.SLOPE_MODE:
                self.run_slope_mode()

            elif self.mode == self.RELOCALIZE:
                self.run_relocalize()

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("slope_mode_controller")
    node = SlopeModeController()
    node.spin()
