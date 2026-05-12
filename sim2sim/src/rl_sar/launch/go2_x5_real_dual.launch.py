# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    network_interface = LaunchConfiguration("network_interface")
    start_arm_bridge = LaunchConfiguration("start_arm_bridge")
    arm_model = LaunchConfiguration("arm_model")
    arm_interface_name = LaunchConfiguration("arm_interface_name")
    arm_urdf_path = LaunchConfiguration("arm_urdf_path")
    arm_joint_count = LaunchConfiguration("arm_joint_count")
    arm_cmd_topic = LaunchConfiguration("arm_cmd_topic")
    arm_state_topic = LaunchConfiguration("arm_state_topic")
    arm_publish_rate_hz = LaunchConfiguration("arm_publish_rate_hz")
    arm_command_speed = LaunchConfiguration("arm_command_speed")
    arm_cmd_timeout_sec = LaunchConfiguration("arm_cmd_timeout_sec")
    arm_dry_run = LaunchConfiguration("arm_dry_run")
    arm_sdk_root = LaunchConfiguration("arm_sdk_root")
    arm_sdk_python_path = LaunchConfiguration("arm_sdk_python_path")
    arm_sdk_lib_path = LaunchConfiguration("arm_sdk_lib_path")
    arm_require_sdk = LaunchConfiguration("arm_require_sdk")
    arm_require_initial_state = LaunchConfiguration("arm_require_initial_state")
    arm_enable_background_send_recv = LaunchConfiguration("arm_enable_background_send_recv")
    arm_controller_dt = LaunchConfiguration("arm_controller_dt")
    arm_init_to_home = LaunchConfiguration("arm_init_to_home")

    arm_bridge_node = Node(
        package="rl_sar",
        executable="arx_x5_bridge.py",
        name="arx_x5_bridge",
        output="screen",
        condition=IfCondition(start_arm_bridge),
        parameters=[
            {
                "model": arm_model,
                "interface_name": arm_interface_name,
                "urdf_path": arm_urdf_path,
                "joint_count": arm_joint_count,
                "cmd_topic": arm_cmd_topic,
                "state_topic": arm_state_topic,
                "publish_rate_hz": arm_publish_rate_hz,
                "command_speed": arm_command_speed,
                "cmd_timeout_sec": arm_cmd_timeout_sec,
                "dry_run": arm_dry_run,
                "sdk_root": arm_sdk_root,
                "sdk_python_path": arm_sdk_python_path,
                "sdk_lib_path": arm_sdk_lib_path,
                "require_sdk": arm_require_sdk,
                "require_initial_state": arm_require_initial_state,
                "enable_background_send_recv": arm_enable_background_send_recv,
                "controller_dt": arm_controller_dt,
                "init_to_home": arm_init_to_home,
            }
        ],
    )

    go2_x5_node = Node(
        package="rl_sar",
        executable="rl_real_go2_x5",
        name="rl_real_go2_x5",
        output="screen",
        arguments=[network_interface],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("network_interface", default_value="eth0"),
            DeclareLaunchArgument("start_arm_bridge", default_value="true"),
            DeclareLaunchArgument("arm_model", default_value="X5"),
            DeclareLaunchArgument("arm_interface_name", default_value="can0"),
            DeclareLaunchArgument("arm_urdf_path", default_value=""),
            DeclareLaunchArgument("arm_joint_count", default_value="6"),
            DeclareLaunchArgument("arm_cmd_topic", default_value="/arx_x5/joint_cmd"),
            DeclareLaunchArgument("arm_state_topic", default_value="/arx_x5/joint_state"),
            DeclareLaunchArgument("arm_publish_rate_hz", default_value="200.0"),
            DeclareLaunchArgument("arm_command_speed", default_value="0.4"),
            DeclareLaunchArgument("arm_cmd_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("arm_dry_run", default_value="false"),
            DeclareLaunchArgument("arm_sdk_root", default_value=EnvironmentVariable("ARX5_SDK_ROOT", default_value="")),
            DeclareLaunchArgument("arm_sdk_python_path", default_value=EnvironmentVariable("ARX5_SDK_PYTHON_PATH", default_value="")),
            DeclareLaunchArgument("arm_sdk_lib_path", default_value=EnvironmentVariable("ARX5_SDK_LIB_PATH", default_value="")),
            DeclareLaunchArgument("arm_require_sdk", default_value="true"),
            DeclareLaunchArgument("arm_require_initial_state", default_value="true"),
            DeclareLaunchArgument("arm_enable_background_send_recv", default_value="true"),
            DeclareLaunchArgument("arm_controller_dt", default_value="0.002"),
            DeclareLaunchArgument("arm_init_to_home", default_value="false"),
            arm_bridge_node,
            go2_x5_node,
        ]
    )
