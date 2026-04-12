include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- Frame configuration
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "base_link",
  odom_frame = "odom",

  -- Publish local odometry frame and project published frame to 2D
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  -- Enable pose extrapolator for smoother pose prediction under asynchronous sensor inputs
  use_pose_extrapolator = true,

  -- Sensor inputs
  use_odometry = true,
  use_nav_sat = true,
  use_landmarks = false,

  -- Publish estimated transforms to the TF tree
  publish_to_tf = true,

  -- Sensor configuration
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,

  -- Timing configuration
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  -- Sampling ratios
  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

-- Use 2D trajectory builder
MAP_BUILDER.use_trajectory_builder_2d = true

-- Enable IMU data for 2D SLAM
TRAJECTORY_BUILDER_2D.use_imu_data = true

-- Valid range of laser measurements
TRAJECTORY_BUILDER_2D.min_range = 0.2
TRAJECTORY_BUILDER_2D.max_range = 20.0

-- Conservative free-space ray length for missing returns
-- This reduces overly aggressive clearing in unknown regions
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 2.0

-- Process each incoming scan immediately for low-latency updates
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

-- Number of range data inserted into each submap
-- Larger values improve submap stability and reduce sensitivity to noise
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 1000

-- Enable online correlative scan matching
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true

-- Search window in translation
-- A smaller window relies more on the initial pose estimate
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1

-- Search window in rotation
-- This allows greater tolerance to heading estimation errors
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(35.0)

-- Penalize translational deviation from the predicted pose
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 20.0

-- Penalize rotational deviation from the predicted pose
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1.0

-- Robust loss for pose graph optimization
-- Reduces the influence of outlier constraints
POSE_GRAPH.optimization_problem.huber_scale = 1e2

-- Run pose graph optimization every N nodes
POSE_GRAPH.optimize_every_n_nodes = 5

-- Minimum score required to accept a scan-to-submap constraint
POSE_GRAPH.constraint_builder.min_score = 0.65

return options
