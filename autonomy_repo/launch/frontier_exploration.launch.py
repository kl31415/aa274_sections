#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value = "true"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments = {
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz"
                        ]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            Node(
                executable = "rviz_goal_relay.py",
                package = "asl_tb3_lib",
                parameters = [
                    {"output_channel": "/cmd_nav"},
                ],
            ),
            Node(
                executable = "state_publisher.py",
                package = "asl_tb3_lib",
            ),
            Node(
                executable = "navigator.py",
                package = "autonomy_repo",
                parameters = [
                    {"use_sim_time": use_sim_time},
                ],
            ),
            TimerAction(
                period = 5.0,
                actions=[
                    Node(
                        executable = "frontier_exploration.py",
                        package = "autonomy_repo",
                        parameters = [
                            {"use_sim_time": use_sim_time}
                        ],
                    )
                ]
            ),
        ]
    )
