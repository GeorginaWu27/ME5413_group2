# ME5413_Final_Project_group_2

> Group Members: [WU ZIHAN](https://github.com/GeorginaWu27), [FENG YIXUAN](https://github.com/yixuanfeng15-lgtm), [QIAN YIWEI](https://github.com/qian-yiwei), and [WU YIFEI](https://github.com/wyffei)
> 
> Please find the main project at: https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project
>
> Github link for this repo: https://github.com/GeorginaWu27/ME5413_group2

## Tasks

![task_image](src/me5413_world/media/overview2526.png)

* Please ensure the objects have been successfully generated at the start of each run.
* On the lower floor, count the number of occurrences of each type of numbered box (e.g. box 1, 2, 3, 4; the box numbers are randomly generated).
* Unblock the exit from the lower level by publishing a `true` message (`std_msgs/Bool`) to the `/cmd_unblock` topic to remove the orange barrel.
  * Note that the exit can only be unblocked once, and the unblocked time lasts for 10 s.
* Exit the lower level.
* Go up the ramp to the upper level.
* Navigate past the first two corridors.
* At the wall with the two gaps, an orange traffic cone will be randomly placed in one of them, closing the entrance; use the other door to navigate into the upper-floor main room.
* Avoid the simulated person (represented by the moving red cylinder).
* Finally, stop the Jackal in the room containing the numbered box with the least number of occurrences.
* Please do not use any ground-truth topics such as `/gazebo/ground_truth/state` or `/box_odom`.

## Dependencies

* System Requirements:
  * Ubuntu 20.04
  * ROS Noetic
  * C++11 and above
  * CMake 3.0.2 and above
  * Python 3
* This repo depends on the following standard ROS packages:
  * `roscpp`
  * `rospy`
  * `rviz`
  * `std_msgs`
  * `nav_msgs`
  * `geometry_msgs`
  * `sensor_msgs`
  * `visualization_msgs`
  * `tf`
  * `tf2`
  * `tf2_ros`
  * `tf2_geometry_msgs`
  * `pluginlib`
  * `map_server`
  * `gazebo_ros`
  * `jsk_rviz_plugins`
  * `jackal_gazebo`
  * `jackal_navigation`
  * `velodyne_simulator`
  * `teleop_twist_keyboard`
  * `move_base`
  * `costmap_2d`
  * `robot_localization`
* In addition, this repo also requires the following navigation planner plugins:
  * `teb_local_planner`
  * `global_planner`
* Python packages required by the perception / evaluation scripts:
  * `numpy`
  * `opencv-python`
  * `easyocr`
  * `scipy`
  * `matplotlib`
* And this [gazebo_models](https://github.com/osrf/gazebo_models) repository.

## Installation

This repo is a ROS workspace containing the following ROS packages:

* `interactive_tools`: customized tools to interact with Gazebo and the robot
* `jackal_description`: modified Jackal robot model descriptions
* `me5413_world`: main package containing the Gazebo world, maps, perception nodes, and launch files
* `teleop_twist_keyboard`: Use keyboard to manually control the Jackal robot
* `third_party`: mapping package for cartographer
* `me5413_navigation`: navigation package for localization, planning, waypoint patrol, and evaluation

```bash
# Clone this repo
cd ~
git clone https://github.com/GeorginaWu27/ME5413_group2.git
cd ME5413_group2
```

* Download the bags file from https://drive.google.com/file/d/1NHFYNkYMH1d5Fngifpc0DH4SxrK8I-Q0/view?usp=sharing, unzip it and put it under the same ME5413_group2 folder

```
# Install all dependencies
rosdep install --from-paths src --ignore-src -r -y

# Sometimes there might be missing dependencies such as sensor drivers
# and navigation plugins for the simulation
sudo apt install -y \
  ros-noetic-sick-tim \
  ros-noetic-lms1xx \
  ros-noetic-velodyne-description \
  ros-noetic-pointgrey-camera-description \
  ros-noetic-jackal-control \
  ros-noetic-teb-local-planner \
  ros-noetic-global-planner \
  ros-noetic-cv-bridge \
  ros-noetic-rviz-imu-plugin \
  ros-noetic-jsk-rviz-plugins

# Install required Python packages
pip3 install numpy opencv-python easyocr scipy matplotlib

# Build the workspace
catkin_make_isolated

# Source
source devel_isolated/setup.bash
```

To properly load the Gazebo world, you will need to have the necessary model files in the `~/.gazebo/models/` directory.

There are two sources of models needed:

* [Gazebo official models](https://github.com/osrf/gazebo_models)

  ```bash
  # Create the destination directory
  cd ~
  mkdir -p ~/.gazebo/models

  # Clone the official Gazebo models repo
  git clone https://github.com/osrf/gazebo_models.git

  # Copy the models into the ~/.gazebo/models directory
  cp -r ~/gazebo_models/* ~/.gazebo/models/
  ```

* [Provided customized models](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project/tree/main/src/me5413_world/models)

  ```bash
  # Copy the customized models into the ~/.gazebo/models directory
  cp -r ~/ME5413_Final_Project/src/me5413_world/models/* ~/.gazebo/models/
  ```

## Usage

### 0. Source environment

```bash
# Workspace
cd ~/ME5413_group2

# Build
catkin_make_isolated

# Source
source devel_isolated/setup.bash

#Remove install_isolated

```

### 1. Load Gazebo world

This command launches Gazebo with the project world and random obstacles / task objects.

```bash
# Launch Gazebo world together with the robot
source devel_isolated/setup.bash
roslaunch me5413_world world.launch
```

This launch file will:

* load the project world
* spawn the Jackal robot
* load the destination configuration
* automatically publish a respawn command once at startup to generate the randomized objects

### 2. Mapping

Mapping is optional for the final autonomous run because a map file is already provided at (https://drive.google.com/file/d/16jMduRXgbkZkFMURzpQAJd0ris-tp8EG/view?usp=sharing).

If you want to build or update the map manually, after launching **Step 1**, open a new terminal and run:

**Steps to build cartographer**
```bash
cd ~/ME5413_Final_Project
rosdep install --from-paths src --ignore-src -r -y

cd src/third_party/cartographer/scripts
#You may need to update the paths in the two .sh scripts to match your actual absolute paths.
./install_abseil.sh

#Cartographer use catkin_make_isolated to compile
catkin_make_isolated --install
#Remove install_isolated
```

You can either drive the robot in Gazebo using the keyboard (see the terminal output for controls) to explore and build the map, or use a pre-recorded ROS bag for mapping.

**Manually**
```bash
#In terminal 1:Launch the world
cd ~/ME5413_Final_Project
source devel_isolated/setup.bash
roslaunch me5413_world world.launch

#In terminal 2:Launch keyboard control
cd ~/ME5413_Final_Project
source devel_isolated/setup.bash
roslaunch me5413_world manual.launch

#In terminal 3:Launch cartographer
cd ~/ME5413_Final_Project
source devel_isolated/setup.bash
roslaunch me5413_world cartographer_2d_IMU_ODOM_LI.launch
```

**Rosbag**
```bash
#In terminal 1:
roscore

#In terminal 2:
cd your rosbag workspace
rosparam set use_sim_time true

#In terminal 3:
roslaunch me5413_world cartographer_2d_IMU_ODOM_LI.launch

#In terminal 2:
rosbag play <bag's name>  --clock

```

After finishing mapping, run the following command in the terminal to save the map:
```bash
# Save the map as `my_map` in the `maps/` folder
roscd me5413_world/maps/
rosrun map_server map_saver -f my_map map:=/map
```
### 3. Navigation

After launching **Step 1**, open a second terminal and run:

```bash
cd ~/ME5413_Final_Project
source devel_isolated/setup.bash
roslaunch me5413_world navigation.launch
```

The navigation stack will start, including:

* `map_server`
* `amcl`
* `move_base`
* slope mode controller
* waypoint patrol navigation
* RViz

Navigation module will also trigger perception module, including:

* lower-floor box counting perception
* second-floor perception switcher

The robot will then execute the task autonomously using the prepared map and world setup.

### 4. Evaluation for localization

```bash
# Estimate tf between world frame and map frame manually
rosrun me5413_navigation estimate_map_world_tf.py

# Evaluate the ground truth of the robot and AMCL using the manually estimated tf
# Assume the pose of the world-frame origin in the map frame is:
# (20.593895, 11.758865, 0.197229)
rosrun me5413_navigation amcl_truth_eval.py \
  _tx:=20.593895 \
  _ty:=11.758865 \
  _theta:=0.197229 \
  _save_path:=/tmp/amcl_eval.csv

# Visualize the evaluation result
python3 ~/ME5413_Final_Project/src/me5413_navigation/scripts/amcl_eval_plot.py \
  --csv /tmp/amcl_eval.csv \
  --outdir /tmp/amcl_plots \
  --normal-only
```

If needed, copy the generated results to your project directory:

```bash
cp /tmp/amcl_eval.csv ~/ME5413_Final_Project/
cp /tmp/amcl_plots/* ~/ME5413_Final_Project/plots/
```

## Notes

* The main autonomous pipeline is launched by `world.launch` + `navigation.launch`.
* `navigation.launch` also starts the perception modules used for box counting and second-floor room selection.
* The perception node relies on OCR and OpenCV-related Python dependencies, so please ensure the Python packages are installed correctly before running.
* If your workspace path is different, update any hard-coded file paths in your own scripts accordingly.

## License

The [ME5413_Final_Project](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project) is released under the [MIT License](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project/blob/main/LICENSE).



