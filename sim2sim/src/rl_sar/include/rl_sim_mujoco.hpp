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
#include <memory>
#include <atomic>
#include <mutex>
#include <array>

#if defined(USE_ROS1)
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#elif defined(USE_ROS2)
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#endif

#include <mujoco/mujoco.h>
#include "joystick.hh"
#include "mujoco_utils.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Button
{
public:
    Button() {}

    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

    std::unique_ptr<mj::Simulate> sim;
    static RL_Sim* instance;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();
    void ResetEpisode();
    void SetInitialPose();
    void InitMujocoMappings();
    void UpdateArmJointIndices();
    void RefreshArmParamsFromConfig();
    void ApplyArmHoldPose(const std::vector<float>& pose, const char* reason);
    void PublishArmCommand(const std::vector<float>& pose, const char* reason);
    void ValidateLoadedPolicyConfig() override;
    void ResetSimRuntimeState() override;
    void ValidateMujocoActuatorRanges() const;
#if defined(USE_ROS1)
    void ArmCommandCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);
#elif defined(USE_ROS2)
    void ArmCommandCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
#endif

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // mujoco
    mjData *mj_data;
    mjModel *mj_model;
    std::string scene_name;

    // MuJoCo sensor/actuator mappings by joint name (robust to ordering)
    std::vector<int> sensor_qpos_adr;
    std::vector<int> sensor_qvel_adr;
    std::vector<int> sensor_tau_adr;
    std::vector<int> actuator_id;
    int sensor_imu_quat_adr = -1;
    int sensor_imu_gyro_adr = -1;
    int sensor_frame_vel_adr = -1;

    // arm command (embedded)
    int arm_command_size = 0;
    bool arm_hold_enabled = true;
    std::atomic<bool> arm_use_embedded_command{false};
    std::string arm_joint_command_topic;
    std::vector<float> arm_hold_position;
    std::vector<float> arm_embedded_command;
    std::vector<float> arm_output_pos_filtered;
    bool arm_output_pos_initialized = false;
    float arm_output_filter_alpha = 0.1f;
    bool arm_hold_pose_applied = false;
    std::mutex arm_command_mutex;
    std::vector<int> arm_joint_indices;
    bool arm_pose_active = false;
    bool base_lock_active = false;
    std::array<mjtNum, 7> base_lock_qpos{};

#if defined(USE_ROS1)
    ros::NodeHandle ros_node;
    ros::Publisher arm_joint_command_publisher;
    ros::Subscriber arm_joint_command_subscriber;
#elif defined(USE_ROS2)
    std::shared_ptr<rclcpp::Node> ros2_node;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr arm_joint_command_publisher;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr arm_joint_command_subscriber;
    bool ros2_initialized = false;
#endif

    // joystick
    std::unique_ptr<Joystick> sys_js;
    JoystickEvent sys_js_event;

    Button sys_js_button[20];
    int sys_js_axis[10] = {0};
    bool sys_js_active = false;
    float axis_deadzone = 0.05f;
    int sys_js_max_value = (1 << (16 - 1));
    void SetupSysJoystick(const std::string& device, int bits);
    void GetSysJoystick();

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);
};

#endif // RL_SIM_HPP
