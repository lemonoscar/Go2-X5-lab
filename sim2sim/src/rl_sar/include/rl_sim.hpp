/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_all.hpp"

#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <mutex>

#if defined(USE_ROS1)
#include <ros/ros.h>
#include "std_srvs/Empty.h"
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <gazebo_msgs/ModelStates.h>
#include "robot_msgs/MotorCommand.h"
#include "robot_msgs/MotorState.h"
#elif defined(USE_ROS2)
#include "robot_msgs/msg/robot_command.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include <gazebo_msgs/msg/model_states.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

#if defined(USE_ROS2)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();
    void ResetEpisode();
    void ValidateLoadedPolicyConfig() override;
    void ResetSimRuntimeState() override;

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // ros interface
    std::string ros_namespace;
#if defined(USE_ROS1)
    geometry_msgs::Twist vel;
    geometry_msgs::Pose pose;
    geometry_msgs::Twist cmd_vel;
    geometry_msgs::Twist cmd_vel_filtered;
    bool cmd_vel_has_filtered = false;
    sensor_msgs::Joy joy_msg;
    ros::Subscriber model_state_subscriber;
    ros::Subscriber cmd_vel_subscriber;
    ros::Subscriber joy_subscriber;
    ros::ServiceClient gazebo_pause_physics_client;
    ros::ServiceClient gazebo_unpause_physics_client;
    ros::ServiceClient gazebo_reset_world_client;
    std::map<std::string, ros::Publisher> joint_publishers;
    std::map<std::string, ros::Subscriber> joint_subscribers;
    std::vector<robot_msgs::MotorCommand> joint_publishers_commands;
    void ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg);
    void JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name);
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
    void JoyCallback(const sensor_msgs::Joy::ConstPtr &msg);
#elif defined(USE_ROS2)
    sensor_msgs::msg::Imu gazebo_imu;
    geometry_msgs::msg::Twist gazebo_model_twist;
    geometry_msgs::msg::Twist cmd_vel;
    geometry_msgs::msg::Twist cmd_vel_filtered;
    bool cmd_vel_has_filtered = false;
    sensor_msgs::msg::Joy joy_msg;
    robot_msgs::msg::RobotCommand robot_command_publisher_msg;
    robot_msgs::msg::RobotState robot_state_subscriber_msg;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gazebo_imu_subscriber;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_state_subscriber;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_state_subscriber_compat;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr height_scan_subscriber;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr arm_joint_command_subscriber;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_pause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_unpause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_reset_world_client;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;
    void GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void ModelStatesCallback(const gazebo_msgs::msg::ModelStates::SharedPtr msg);
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void HeightScanCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void ArmJointCommandCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
#endif
    float cmd_vel_alpha = 0.2f;

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);

    // height scan and arm command
    std::vector<float> height_scan_latest;
    std::vector<float> arm_joint_command_latest;
    std::mutex height_scan_mutex;
    std::mutex arm_command_mutex;
    std::mutex state_msg_mutex;
    std::mutex robot_state_mutex;
    std::mutex cmd_vel_mutex;
    int height_scan_width = 0;
    int height_scan_height = 0;
    float height_scan_offset = 0.5f;
    float height_scan_clip = 1.0f;
    bool height_scan_flip_x = false;
    bool height_scan_flip_y = false;
    std::string height_scan_topic;
    std::string arm_joint_command_topic;

    // arm command sequence (keyboard-triggered)
    bool arm_sequence_active = false;
    bool arm_sequence_loop = false;
    int arm_sequence_steps = 0;
    int arm_sequence_ticks = 0;
    int arm_sequence_interval_ticks = 0;
    size_t arm_sequence_index = 0;
    std::vector<float> arm_sequence_flat;
    std::vector<float> arm_sequence_current;
    int arm_command_size = 6;
    bool arm_hold_enabled = true;
    bool arm_pose_active = false;
    std::vector<float> arm_hold_position;

    void StartArmSequence(bool loop);
    void StopArmSequence();
    void UpdateArmSequence();
    bool ArmCommandDifferent(const std::vector<float>& a, const std::vector<float>& b) const;

    // arm command smoothing
    int arm_command_smoothing_ticks = 0;
    int arm_command_smoothing_counter = 0;
    std::vector<float> arm_command_smoothing_start;
    std::vector<float> arm_command_smoothing_target;
    std::vector<float> arm_command_smoothed;
    bool arm_command_initialized = false;
    std::vector<float> arm_output_pos_filtered;
    bool arm_output_pos_initialized = false;
    float arm_output_filter_alpha = 0.1f;

    // output smoothing
    std::vector<float> output_dof_pos_filtered;
    bool output_dof_pos_initialized = false;
};

#endif // RL_SIM_HPP
