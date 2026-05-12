/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_go2_x5.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>

RL_Real_Go2X5::RL_Real_Go2X5(int argc, char **argv)
{
    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = "go2_x5";
    this->ReadYaml(this->robot_name, "base.yaml");
    this->InitializeArmCommandState();
    this->InitializeArmChannelConfig();
    this->ValidateJointMappingOrThrow("base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

#if defined(USE_ROS1) && defined(USE_ROS)
    this->ros1_nh = std::make_shared<ros::NodeHandle>();
    this->cmd_vel_subscriber = this->ros1_nh->subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real_Go2X5::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_go2_x5_node");
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) { this->CmdvelCallback(msg); });
#endif
    this->SetupArmCommandSubscriber();
    this->SetupArmBridgeInterface();

    // init robot
    this->InitLowCmd();
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // create lowcmd publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    this->lowcmd_publisher->InitChannel();
    // create lowstate subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real_Go2X5::LowStateMessageHandler, this, std::placeholders::_1), 1);
    // create joystick subscriber
    this->joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    this->joystick_subscriber->InitChannel(std::bind(&RL_Real_Go2X5::JoystickHandler, this, std::placeholders::_1), 1);

    // init MotionSwitcherClient
    this->msc.SetTimeout(10.0f);
    this->msc.Init();
    // Shut down motion control-related service
    while (this->QueryMotionStatus())
    {
        std::cout << "Try to deactivate the motion control-related service." << std::endl;
        int32_t ret = this->msc.ReleaseMode();
        if (ret == 0)
        {
            std::cout << "ReleaseMode succeeded." << std::endl;
        }
        else
        {
            std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
        }
        sleep(1);
    }

    std::cout << LOGGER::INFO << "Real deploy target: go2_x5" << std::endl;
    std::cout << LOGGER::INFO << "arm_joint_command_topic: " << this->arm_joint_command_topic
              << ", arm_hold_enabled: " << (this->arm_hold_enabled ? "true" : "false") << std::endl;

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real_Go2X5::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real_Go2X5::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real_Go2X5::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real_Go2X5::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real_Go2X5::~RL_Real_Go2X5()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real_Go2X5 exit" << std::endl;
}

void RL_Real_Go2X5::InitializeArmCommandState()
{
    this->cmd_vel_alpha = this->params.Get<float>("cmd_vel_alpha", this->cmd_vel_alpha);
    this->arm_command_size = this->params.Get<int>("arm_command_size", 0);
    this->arm_joint_command_topic = this->params.Get<std::string>("arm_joint_command_topic", "/arm_joint_pos_cmd");
    this->arm_hold_enabled = this->params.Get<bool>("arm_hold_enabled", true);

    const float step_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    const float smooth_time = this->params.Get<float>("arm_command_smoothing_time", 0.2f);
    if (step_dt > 0.0f)
    {
        this->arm_command_smoothing_ticks = std::max(0, static_cast<int>(std::lround(smooth_time / step_dt)));
    }
    else
    {
        this->arm_command_smoothing_ticks = 0;
    }
    this->arm_command_smoothing_counter = 0;

    if (this->arm_command_size <= 0)
    {
        this->arm_joint_command_latest.clear();
        this->arm_hold_position.clear();
        this->arm_command_smoothing_start.clear();
        this->arm_command_smoothing_target.clear();
        this->arm_command_smoothed.clear();
        this->arm_command_initialized = false;
        return;
    }

    this->arm_hold_position.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
    const auto hold_pose = this->params.Get<std::vector<float>>("arm_hold_pose", {});
    if (hold_pose.size() == static_cast<size_t>(this->arm_command_size))
    {
        this->arm_hold_position = hold_pose;
    }
    else
    {
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos", {});
        if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            this->arm_hold_position.assign(
                default_pos.begin() + static_cast<long>(arm_start),
                default_pos.end()
            );
        }
    }

    this->arm_joint_command_latest = this->arm_hold_position;
    this->arm_command_smoothing_start = this->arm_hold_position;
    this->arm_command_smoothing_target = this->arm_hold_position;
    this->arm_command_smoothed = this->arm_hold_position;
    this->arm_command_initialized = true;
}

void RL_Real_Go2X5::InitializeArmChannelConfig()
{
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    this->arm_joint_count = this->params.Get<int>("arm_joint_count", this->arm_command_size > 0 ? this->arm_command_size : 6);
    this->arm_joint_start_index = this->params.Get<int>("arm_joint_start_index", std::max(0, num_dofs - this->arm_joint_count));

    if (this->arm_joint_start_index < 0)
    {
        this->arm_joint_start_index = 0;
    }
    if (this->arm_joint_start_index > num_dofs)
    {
        this->arm_joint_start_index = num_dofs;
    }
    if (this->arm_joint_count < 0)
    {
        this->arm_joint_count = 0;
    }
    if (this->arm_joint_start_index + this->arm_joint_count > num_dofs)
    {
        this->arm_joint_count = std::max(0, num_dofs - this->arm_joint_start_index);
    }

    this->arm_control_mode = this->params.Get<std::string>("arm_control_mode", "unitree");
    std::transform(this->arm_control_mode.begin(), this->arm_control_mode.end(), this->arm_control_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    this->arm_split_control_enabled =
        (this->arm_control_mode == "split" || this->arm_control_mode == "arx_x5" || this->arm_control_mode == "bridge");

#if !defined(USE_ARX_X5_SDK)
    if (this->arm_control_mode == "arx_x5")
    {
        std::cout << LOGGER::WARNING
                  << "arm_control_mode=arx_x5 but binary was built without USE_ARX_X5_SDK. "
                  << "Falling back to bridge transport." << std::endl;
    }
#endif

    this->arm_bridge_cmd_topic = this->params.Get<std::string>("arm_bridge_cmd_topic", "/arx_x5/joint_cmd");
    this->arm_bridge_state_topic = this->params.Get<std::string>("arm_bridge_state_topic", "/arx_x5/joint_state");
    this->arm_bridge_require_state = this->params.Get<bool>("arm_bridge_require_state", this->arm_split_control_enabled);
    this->arm_bridge_state_timeout_sec = this->params.Get<float>("arm_bridge_state_timeout_sec", 0.25f);
    if (this->arm_bridge_state_timeout_sec < 0.0f)
    {
        this->arm_bridge_state_timeout_sec = 0.0f;
    }
    this->arm_bridge_state_timeout_warned = false;

    if (this->arm_external_state_q.size() != static_cast<size_t>(this->arm_joint_count))
    {
        this->arm_external_state_q.assign(static_cast<size_t>(this->arm_joint_count), 0.0f);
        this->arm_external_state_dq.assign(static_cast<size_t>(this->arm_joint_count), 0.0f);
        this->arm_external_state_tau.assign(static_cast<size_t>(this->arm_joint_count), 0.0f);
        this->arm_bridge_state_valid = false;
    }
    if (this->arm_external_shadow_q.size() != static_cast<size_t>(this->arm_joint_count))
    {
        this->arm_external_shadow_q.assign(static_cast<size_t>(this->arm_joint_count), 0.0f);
        this->arm_external_shadow_dq.assign(static_cast<size_t>(this->arm_joint_count), 0.0f);
        const int expected_arm_size = std::max(0, this->arm_command_size);
        if (expected_arm_size > 0 && this->arm_hold_position.size() == static_cast<size_t>(expected_arm_size))
        {
            const int count = std::min(this->arm_joint_count, expected_arm_size);
            for (int i = 0; i < count; ++i)
            {
                this->arm_external_shadow_q[static_cast<size_t>(i)] = this->arm_hold_position[static_cast<size_t>(i)];
            }
        }
    }

    std::cout << LOGGER::INFO << "Arm control mode: " << this->arm_control_mode
              << " (start=" << this->arm_joint_start_index
              << ", count=" << this->arm_joint_count << ")" << std::endl;
    if (this->arm_split_control_enabled)
    {
        std::cout << LOGGER::INFO << "Arm bridge guard: require_state="
                  << (this->arm_bridge_require_state ? "true" : "false")
                  << ", timeout=" << this->arm_bridge_state_timeout_sec << "s" << std::endl;
    }
}

void RL_Real_Go2X5::ValidateJointMappingOrThrow(const char* stage) const
{
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (num_dofs <= 0)
    {
        throw std::runtime_error(std::string("Invalid num_of_dofs in ") + stage);
    }

    const auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping", {});
    if (joint_mapping.size() != static_cast<size_t>(num_dofs))
    {
        throw std::runtime_error(
            std::string("joint_mapping size mismatch in ") + stage +
            ": expect " + std::to_string(num_dofs) +
            ", got " + std::to_string(joint_mapping.size()));
    }

    std::vector<bool> used(20, false);
    for (size_t i = 0; i < joint_mapping.size(); ++i)
    {
        const int mapped = joint_mapping[i];
        if (mapped < 0 || mapped >= 20)
        {
            throw std::runtime_error(
                std::string("joint_mapping out of range in ") + stage +
                " at index " + std::to_string(i) +
                ", value=" + std::to_string(mapped) + ", expected [0,19]");
        }
        if (used[static_cast<size_t>(mapped)])
        {
            std::cout << LOGGER::WARNING
                      << "Duplicate joint_mapping value " << mapped
                      << " detected in " << stage << "." << std::endl;
        }
        used[static_cast<size_t>(mapped)] = true;
    }

    if (this->arm_joint_start_index < 0 || this->arm_joint_count < 0 ||
        (this->arm_joint_start_index + this->arm_joint_count) > num_dofs)
    {
        throw std::runtime_error(
            std::string("Invalid arm index window in ") + stage +
            ": start=" + std::to_string(this->arm_joint_start_index) +
            ", count=" + std::to_string(this->arm_joint_count) +
            ", num_of_dofs=" + std::to_string(num_dofs));
    }
}

bool RL_Real_Go2X5::IsArmBridgeStateFreshLocked() const
{
    if (!this->arm_bridge_state_valid)
    {
        return false;
    }

    if (this->arm_bridge_state_timeout_sec <= 1e-6f)
    {
        return true;
    }

    const auto age = std::chrono::duration_cast<std::chrono::duration<float>>(
        std::chrono::steady_clock::now() - this->arm_bridge_state_stamp).count();
    return age <= this->arm_bridge_state_timeout_sec;
}

void RL_Real_Go2X5::SetupArmCommandSubscriber()
{
#if !defined(USE_CMAKE) && defined(USE_ROS)
    if (this->arm_joint_command_topic.empty())
    {
        return;
    }
    if (this->arm_joint_command_topic == this->arm_joint_command_topic_active)
    {
        return;
    }
#if defined(USE_ROS1) && defined(USE_ROS)
    if (!this->ros1_nh)
    {
        return;
    }
    this->arm_joint_command_subscriber = this->ros1_nh->subscribe<std_msgs::Float32MultiArray>(
        this->arm_joint_command_topic, 10, &RL_Real_Go2X5::ArmJointCommandCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    if (!this->ros2_node)
    {
        return;
    }
    this->arm_joint_command_subscriber = this->ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
        this->arm_joint_command_topic, rclcpp::SystemDefaultsQoS(),
        [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) { this->ArmJointCommandCallback(msg); });
#endif
    this->arm_joint_command_topic_active = this->arm_joint_command_topic;
    std::cout << LOGGER::INFO << "arm_joint_command_topic subscribed: "
              << this->arm_joint_command_topic_active << std::endl;
#endif
}

void RL_Real_Go2X5::SetupArmBridgeInterface()
{
#if !defined(USE_CMAKE) && defined(USE_ROS)
    if (!this->arm_split_control_enabled || this->arm_joint_count <= 0)
    {
        return;
    }
    if (this->arm_bridge_cmd_topic.empty() || this->arm_bridge_state_topic.empty())
    {
        std::cout << LOGGER::WARNING << "Arm split control enabled but bridge topics are empty." << std::endl;
        return;
    }
#if defined(USE_ROS1) && defined(USE_ROS)
    if (!this->ros1_nh)
    {
        return;
    }
    this->arm_bridge_cmd_publisher =
        this->ros1_nh->advertise<std_msgs::Float32MultiArray>(this->arm_bridge_cmd_topic, 1);
    this->arm_bridge_state_subscriber =
        this->ros1_nh->subscribe<std_msgs::Float32MultiArray>(
            this->arm_bridge_state_topic, 10, &RL_Real_Go2X5::ArmBridgeStateCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    if (!this->ros2_node)
    {
        return;
    }
    this->arm_bridge_cmd_publisher =
        this->ros2_node->create_publisher<std_msgs::msg::Float32MultiArray>(
            this->arm_bridge_cmd_topic, rclcpp::SystemDefaultsQoS());
    this->arm_bridge_state_subscriber =
        this->ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
            this->arm_bridge_state_topic, rclcpp::SystemDefaultsQoS(),
            [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) { this->ArmBridgeStateCallback(msg); });
#endif
    std::cout << LOGGER::INFO << "Arm bridge ready. cmd_topic=" << this->arm_bridge_cmd_topic
              << ", state_topic=" << this->arm_bridge_state_topic << std::endl;
#else
    if (this->arm_split_control_enabled)
    {
        std::cout << LOGGER::WARNING
                  << "Arm split control enabled in non-ROS build. External arm bridge is unavailable." << std::endl;
    }
#endif
}

bool RL_Real_Go2X5::IsArmJointIndex(int idx) const
{
    return idx >= this->arm_joint_start_index &&
           idx < (this->arm_joint_start_index + this->arm_joint_count);
}

void RL_Real_Go2X5::ReadArmStateFromExternal(RobotState<float> *state)
{
    if (!this->arm_split_control_enabled || this->arm_joint_count <= 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->arm_external_state_mutex);
    const bool bridge_state_fresh = this->IsArmBridgeStateFreshLocked();
    const bool use_bridge_state = bridge_state_fresh &&
                                  this->arm_external_state_q.size() >= static_cast<size_t>(this->arm_joint_count);

    if (this->arm_split_control_enabled && this->arm_bridge_require_state && !bridge_state_fresh &&
        !this->arm_bridge_state_timeout_warned)
    {
        this->arm_bridge_state_timeout_warned = true;
        std::cout << LOGGER::WARNING
                  << "Arm bridge state is missing or stale. Using shadow arm state until bridge recovers."
                  << std::endl;
    }
    else if (bridge_state_fresh)
    {
        this->arm_bridge_state_timeout_warned = false;
    }

    for (int i = 0; i < this->arm_joint_count; ++i)
    {
        const int idx = this->arm_joint_start_index + i;
        if (idx < 0 || idx >= static_cast<int>(state->motor_state.q.size()))
        {
            continue;
        }

        if (use_bridge_state)
        {
            state->motor_state.q[idx] = this->arm_external_state_q[static_cast<size_t>(i)];
            state->motor_state.dq[idx] = this->arm_external_state_dq[static_cast<size_t>(i)];
            state->motor_state.tau_est[idx] = this->arm_external_state_tau[static_cast<size_t>(i)];
        }
        else if (this->arm_external_shadow_q.size() >= static_cast<size_t>(this->arm_joint_count))
        {
            state->motor_state.q[idx] = this->arm_external_shadow_q[static_cast<size_t>(i)];
            state->motor_state.dq[idx] = this->arm_external_shadow_dq[static_cast<size_t>(i)];
            state->motor_state.tau_est[idx] = 0.0f;
        }
    }
}

void RL_Real_Go2X5::WriteArmCommandToExternal(const RobotCommand<float> *command)
{
    if (!this->arm_split_control_enabled || this->arm_joint_count <= 0)
    {
        return;
    }

    std::vector<float> arm_q(static_cast<size_t>(this->arm_joint_count), 0.0f);
    std::vector<float> arm_dq(static_cast<size_t>(this->arm_joint_count), 0.0f);
    std::vector<float> arm_kp(static_cast<size_t>(this->arm_joint_count), 0.0f);
    std::vector<float> arm_kd(static_cast<size_t>(this->arm_joint_count), 0.0f);
    std::vector<float> arm_tau(static_cast<size_t>(this->arm_joint_count), 0.0f);
    for (int i = 0; i < this->arm_joint_count; ++i)
    {
        const int idx = this->arm_joint_start_index + i;
        if (idx < 0 || idx >= static_cast<int>(command->motor_command.q.size()))
        {
            continue;
        }
        arm_q[static_cast<size_t>(i)] = command->motor_command.q[idx];
        arm_dq[static_cast<size_t>(i)] = command->motor_command.dq[idx];
        arm_kp[static_cast<size_t>(i)] = command->motor_command.kp[idx];
        arm_kd[static_cast<size_t>(i)] = command->motor_command.kd[idx];
        arm_tau[static_cast<size_t>(i)] = command->motor_command.tau[idx];
    }

    {
        std::lock_guard<std::mutex> lock(this->arm_external_state_mutex);
        this->arm_external_shadow_q = arm_q;
        this->arm_external_shadow_dq = arm_dq;
    }

#if !defined(USE_CMAKE) && defined(USE_ROS)
    if (!this->arm_bridge_cmd_topic.empty())
    {
#if defined(USE_ROS1) && defined(USE_ROS)
        if (this->arm_bridge_cmd_publisher)
        {
            std_msgs::Float32MultiArray msg;
            msg.data.reserve(static_cast<size_t>(this->arm_joint_count) * 5);
            msg.data.insert(msg.data.end(), arm_q.begin(), arm_q.end());
            msg.data.insert(msg.data.end(), arm_dq.begin(), arm_dq.end());
            msg.data.insert(msg.data.end(), arm_kp.begin(), arm_kp.end());
            msg.data.insert(msg.data.end(), arm_kd.begin(), arm_kd.end());
            msg.data.insert(msg.data.end(), arm_tau.begin(), arm_tau.end());
            this->arm_bridge_cmd_publisher.publish(msg);
        }
#elif defined(USE_ROS2) && defined(USE_ROS)
        if (this->arm_bridge_cmd_publisher)
        {
            std_msgs::msg::Float32MultiArray msg;
            msg.data.reserve(static_cast<size_t>(this->arm_joint_count) * 5);
            msg.data.insert(msg.data.end(), arm_q.begin(), arm_q.end());
            msg.data.insert(msg.data.end(), arm_dq.begin(), arm_dq.end());
            msg.data.insert(msg.data.end(), arm_kp.begin(), arm_kp.end());
            msg.data.insert(msg.data.end(), arm_kd.begin(), arm_kd.end());
            msg.data.insert(msg.data.end(), arm_tau.begin(), arm_tau.end());
            this->arm_bridge_cmd_publisher->publish(msg);
        }
#endif
    }
#endif
}

void RL_Real_Go2X5::ApplyArmHold(const std::vector<float>& target, const char* reason)
{
    if (this->arm_command_size <= 0)
    {
        return;
    }

    if (target.size() != static_cast<size_t>(this->arm_command_size))
    {
        std::cout << LOGGER::WARNING << "Ignore arm hold update: target size mismatch" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(this->arm_command_mutex);
    this->arm_hold_position = target;
    this->arm_joint_command_latest = target;
    this->arm_hold_enabled = true;
    if (!this->arm_command_initialized || this->arm_command_smoothed.size() != target.size())
    {
        this->arm_command_smoothing_start = target;
        this->arm_command_smoothing_target = target;
        this->arm_command_smoothed = target;
        this->arm_command_initialized = true;
        this->arm_command_smoothing_counter = this->arm_command_smoothing_ticks;
    }
    else
    {
        this->arm_command_smoothing_start = this->arm_command_smoothed;
        this->arm_command_smoothing_target = target;
        this->arm_command_smoothing_counter = 0;
    }

    std::cout << LOGGER::INFO << reason << " (smooth=" << this->params.Get<float>("arm_command_smoothing_time", 0.2f) << "s)" << std::endl;
}

bool RL_Real_Go2X5::ArmCommandDifferent(const std::vector<float>& a, const std::vector<float>& b) const
{
    if (a.size() != b.size())
    {
        return true;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::fabs(a[i] - b[i]) > 1e-5f)
        {
            return true;
        }
    }
    return false;
}

void RL_Real_Go2X5::GetState(RobotState<float> *state)
{
    if (this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->unitree_joy.components.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joystick.ly();
    this->control.y = -this->joystick.lx();
    this->control.yaw = -this->joystick.rx();

    state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
    state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
    state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
    state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const int mapped = this->params.Get<std::vector<int>>("joint_mapping")[i];
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[mapped].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[mapped].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[mapped].tau_est();
    }

    this->ReadArmStateFromExternal(state);
}

void RL_Real_Go2X5::SetCommand(const RobotCommand<float> *command)
{
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const int mapped = this->params.Get<std::vector<int>>("joint_mapping")[i];
        if (this->arm_split_control_enabled && this->IsArmJointIndex(i))
        {
            // Arm joints are controlled by an external channel in split mode.
            this->unitree_low_command.motor_cmd()[mapped].mode() = 0x00;
            this->unitree_low_command.motor_cmd()[mapped].q() = PosStopF;
            this->unitree_low_command.motor_cmd()[mapped].dq() = VelStopF;
            this->unitree_low_command.motor_cmd()[mapped].kp() = 0.0f;
            this->unitree_low_command.motor_cmd()[mapped].kd() = 0.0f;
            this->unitree_low_command.motor_cmd()[mapped].tau() = 0.0f;
        }
        else
        {
            this->unitree_low_command.motor_cmd()[mapped].mode() = 0x01;
            this->unitree_low_command.motor_cmd()[mapped].q() = command->motor_command.q[i];
            this->unitree_low_command.motor_cmd()[mapped].dq() = command->motor_command.dq[i];
            this->unitree_low_command.motor_cmd()[mapped].kp() = command->motor_command.kp[i];
            this->unitree_low_command.motor_cmd()[mapped].kd() = command->motor_command.kd[i];
            this->unitree_low_command.motor_cmd()[mapped].tau() = command->motor_command.tau[i];
        }
    }

    this->WriteArmCommandToExternal(command);

    this->unitree_low_command.crc() = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(unitree_low_command);
}

void RL_Real_Go2X5::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::Num2)
    {
        std::vector<float> pose = this->params.Get<std::vector<float>>("arm_key_pose", {});
        if (pose.size() < static_cast<size_t>(this->arm_command_size))
        {
            pose = this->params.Get<std::vector<float>>("arm_hold_pose", {});
        }
        if (!pose.empty() && this->arm_command_size > 0)
        {
            if (pose.size() >= static_cast<size_t>(this->arm_command_size))
            {
                pose.resize(static_cast<size_t>(this->arm_command_size));
                this->ApplyArmHold(pose, "Key[2] pressed: arm hold pose");
            }
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (this->control.current_keyboard == Input::Keyboard::Num3)
    {
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos", {});
        if (this->arm_command_size > 0 && default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            std::vector<float> pose(
                default_pos.begin() + static_cast<long>(arm_start),
                default_pos.end()
            );
            this->ApplyArmHold(pose, "Key[3] pressed: arm restore default");
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (this->control.current_keyboard == Input::Keyboard::Num4)
    {
        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            this->arm_hold_enabled = !this->arm_hold_enabled;
            std::cout << LOGGER::INFO << "Key[4] pressed: arm hold "
                      << (this->arm_hold_enabled ? "ON" : "OFF") << std::endl;
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Real_Go2X5::RunModel()
{
    if (!this->rl_init_done)
    {
        this->arm_runtime_params_ready = false;
        return;
    }
    if (!this->arm_runtime_params_ready)
    {
        // Re-sync arm settings after policy config (go2_x5/robot_lab/config.yaml) is loaded.
        this->InitializeArmCommandState();
        this->InitializeArmChannelConfig();
        this->ValidateJointMappingOrThrow("go2_x5/robot_lab/config.yaml");
        this->SetupArmCommandSubscriber();
        this->SetupArmBridgeInterface();
        this->arm_runtime_params_ready = true;
    }

    this->episode_length_buf += 1;
    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    this->obs.commands = {this->control.x, this->control.y, this->control.yaw};

#if !defined(USE_CMAKE) && defined(USE_ROS)
    if (this->control.navigation_mode)
    {
        std::lock_guard<std::mutex> lock(this->cmd_vel_mutex);
        const auto cmd = this->cmd_vel_has_filtered ? this->cmd_vel_filtered : this->cmd_vel;
#if defined(USE_ROS1) && defined(USE_ROS)
        this->obs.commands = {(float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.angular.z};
#elif defined(USE_ROS2) && defined(USE_ROS)
        this->obs.commands = {(float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.angular.z};
#endif
    }
#endif

    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;

    std::vector<float> arm_hold_local;
    bool arm_hold_enabled_local = false;
    if (this->arm_command_size > 0)
    {
        std::vector<float> desired_arm_command;
        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            desired_arm_command = this->arm_joint_command_latest;
            arm_hold_local = this->arm_hold_position;
            arm_hold_enabled_local = this->arm_hold_enabled;
        }

        if (desired_arm_command.size() != static_cast<size_t>(this->arm_command_size))
        {
            desired_arm_command = arm_hold_local;
        }

        if (!desired_arm_command.empty())
        {
            if (!this->arm_command_initialized || this->arm_command_smoothed.size() != desired_arm_command.size())
            {
                this->arm_command_smoothed = desired_arm_command;
                this->arm_command_smoothing_start = desired_arm_command;
                this->arm_command_smoothing_target = desired_arm_command;
                this->arm_command_smoothing_counter = this->arm_command_smoothing_ticks;
                this->arm_command_initialized = true;
            }

            if (this->ArmCommandDifferent(desired_arm_command, this->arm_command_smoothing_target))
            {
                this->arm_command_smoothing_start = this->arm_command_smoothed;
                this->arm_command_smoothing_target = desired_arm_command;
                this->arm_command_smoothing_counter = 0;
            }

            if (this->arm_command_smoothing_ticks <= 1)
            {
                this->arm_command_smoothed = this->arm_command_smoothing_target;
            }
            else
            {
                const int ticks = this->arm_command_smoothing_ticks;
                int c = std::min(this->arm_command_smoothing_counter + 1, ticks);
                const float alpha = static_cast<float>(c) / static_cast<float>(ticks);
                for (int i = 0; i < this->arm_command_size; ++i)
                {
                    const float start = this->arm_command_smoothing_start[static_cast<size_t>(i)];
                    const float target = this->arm_command_smoothing_target[static_cast<size_t>(i)];
                    this->arm_command_smoothed[static_cast<size_t>(i)] = (1.0f - alpha) * start + alpha * target;
                }
                if (this->arm_command_smoothing_counter < ticks)
                {
                    this->arm_command_smoothing_counter += 1;
                }
            }

            this->obs.arm_joint_command = this->arm_command_smoothed;
        }
    }

    this->obs.actions = this->Forward();
    this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

    if (!this->output_dof_pos.empty() && arm_hold_enabled_local && !arm_hold_local.empty())
    {
        const int num_dofs = this->params.Get<int>("num_of_dofs");
        if (this->arm_command_size > 0 && num_dofs >= this->arm_command_size)
        {
            const int arm_start = num_dofs - this->arm_command_size;
            for (int i = 0; i < this->arm_command_size; ++i)
            {
                const size_t idx = static_cast<size_t>(arm_start + i);
                if (idx < this->output_dof_pos.size() && i < static_cast<int>(arm_hold_local.size()))
                {
                    this->output_dof_pos[idx] = arm_hold_local[static_cast<size_t>(i)];
                }
                if (idx < this->output_dof_vel.size())
                {
                    this->output_dof_vel[idx] = 0.0f;
                }
            }
        }
    }

    if (!this->output_dof_pos.empty() && this->arm_split_control_enabled && this->arm_bridge_require_state)
    {
        bool bridge_state_fresh = false;
        {
            std::lock_guard<std::mutex> lock(this->arm_external_state_mutex);
            bridge_state_fresh = this->IsArmBridgeStateFreshLocked();
        }
        if (!bridge_state_fresh && !arm_hold_local.empty() && this->arm_command_size > 0)
        {
            const int num_dofs = this->params.Get<int>("num_of_dofs");
            const int arm_start = std::max(0, num_dofs - this->arm_command_size);
            for (int i = 0; i < this->arm_command_size; ++i)
            {
                const size_t idx = static_cast<size_t>(arm_start + i);
                if (idx < this->output_dof_pos.size() && i < static_cast<int>(arm_hold_local.size()))
                {
                    this->output_dof_pos[idx] = arm_hold_local[static_cast<size_t>(i)];
                }
                if (idx < this->output_dof_vel.size())
                {
                    this->output_dof_vel[idx] = 0.0f;
                }
            }
        }
    }

    auto build_arm_tracking_gains = [&](std::vector<float>& kp_out, std::vector<float>& kd_out)
    {
        if (!(arm_hold_enabled_local && !arm_hold_local.empty() && this->arm_command_size > 0))
        {
            return;
        }

        kp_out = this->params.Get<std::vector<float>>("rl_kp");
        kd_out = this->params.Get<std::vector<float>>("rl_kd");
        if (kp_out.empty() || kd_out.empty())
        {
            kp_out.clear();
            kd_out.clear();
            return;
        }

        const auto fixed_kp = this->params.Get<std::vector<float>>("fixed_kp");
        const auto fixed_kd = this->params.Get<std::vector<float>>("fixed_kd");
        const auto arm_track_kp = this->params.Get<std::vector<float>>("arm_track_kp", {});
        const auto arm_track_kd = this->params.Get<std::vector<float>>("arm_track_kd", {});
        const bool has_arm_track_kp = arm_track_kp.size() == static_cast<size_t>(this->arm_command_size);
        const bool has_arm_track_kd = arm_track_kd.size() == static_cast<size_t>(this->arm_command_size);
        const int num_dofs = this->params.Get<int>("num_of_dofs");
        const int arm_start = std::max(0, num_dofs - this->arm_command_size);

        for (int i = 0; i < this->arm_command_size; ++i)
        {
            const int idx = arm_start + i;
            if (idx >= static_cast<int>(kp_out.size()) || idx >= static_cast<int>(kd_out.size()))
            {
                continue;
            }

            if (has_arm_track_kp)
            {
                kp_out[static_cast<size_t>(idx)] = arm_track_kp[static_cast<size_t>(i)];
            }
            else if (idx < static_cast<int>(fixed_kp.size()))
            {
                kp_out[static_cast<size_t>(idx)] =
                    std::max(kp_out[static_cast<size_t>(idx)], fixed_kp[static_cast<size_t>(idx)]);
            }

            if (has_arm_track_kd)
            {
                kd_out[static_cast<size_t>(idx)] = arm_track_kd[static_cast<size_t>(i)];
            }
            else if (idx < static_cast<int>(fixed_kd.size()))
            {
                kd_out[static_cast<size_t>(idx)] =
                    std::max(kd_out[static_cast<size_t>(idx)], fixed_kd[static_cast<size_t>(idx)]);
            }
        }
    };

    if (!this->output_dof_pos.empty() || !this->output_dof_vel.empty() || !this->output_dof_tau.empty())
    {
        RLCommandOutput output_cmd;
        output_cmd.pos = this->output_dof_pos;
        output_cmd.vel = this->output_dof_vel;
        output_cmd.tau = this->output_dof_tau;
        build_arm_tracking_gains(output_cmd.kp, output_cmd.kd);
        output_cmd_queue.push(std::move(output_cmd));
    }

    // this->TorqueProtect(this->output_dof_tau);
    // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
    std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
    this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
}

std::vector<float> RL_Real_Go2X5::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Real_Go2X5::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    plt::pause(0.0001);
}

uint32_t RL_Real_Go2X5::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real_Go2X5::InitLowCmd()
{
    this->unitree_low_command.head()[0] = 0xFE;
    this->unitree_low_command.head()[1] = 0xEF;
    this->unitree_low_command.level_flag() = 0xFF;
    this->unitree_low_command.gpio() = 0;

    for (int i = 0; i < 20; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (0x01);
        this->unitree_low_command.motor_cmd()[i].q() = (PosStopF);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (VelStopF);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

int RL_Real_Go2X5::QueryMotionStatus()
{
    std::string robotForm, motionName;
    int motionStatus;
    int32_t ret = this->msc.CheckMode(robotForm, motionName);
    if (ret == 0)
    {
        std::cout << "CheckMode succeeded." << std::endl;
    }
    else
    {
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if (motionName.empty())
    {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        motionStatus = 0;
    }
    else
    {
        std::string serviceName = QueryServiceName(robotForm, motionName);
        std::cout << "Service: " << serviceName << " is activate" << std::endl;
        motionStatus = 1;
    }
    return motionStatus;
}

std::string RL_Real_Go2X5::QueryServiceName(std::string form, std::string name)
{
    if (form == "0")
    {
        if (name == "normal") return "sport_mode";
        if (name == "ai") return "ai_sport";
        if (name == "advanced") return "advanced_sport";
    }
    else
    {
        if (name == "ai-w") return "wheeled_sport(go2W)";
        if (name == "normal-w") return "wheeled_sport(b2W)";
    }
    return "";
}

void RL_Real_Go2X5::LowStateMessageHandler(const void *message)
{
    this->unitree_low_state = *(unitree_go::msg::dds_::LowState_ *)message;
}

void RL_Real_Go2X5::JoystickHandler(const void *message)
{
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    this->unitree_joy.value = joystick.keys();
}

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real_Go2X5::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    std::lock_guard<std::mutex> lock(this->cmd_vel_mutex);
    this->cmd_vel = *msg;
    if (!this->cmd_vel_has_filtered)
    {
        this->cmd_vel_filtered = *msg;
        this->cmd_vel_has_filtered = true;
        return;
    }

    const float a = this->cmd_vel_alpha;
    this->cmd_vel_filtered.linear.x = a * msg->linear.x + (1.0f - a) * this->cmd_vel_filtered.linear.x;
    this->cmd_vel_filtered.linear.y = a * msg->linear.y + (1.0f - a) * this->cmd_vel_filtered.linear.y;
    this->cmd_vel_filtered.linear.z = a * msg->linear.z + (1.0f - a) * this->cmd_vel_filtered.linear.z;
    this->cmd_vel_filtered.angular.x = a * msg->angular.x + (1.0f - a) * this->cmd_vel_filtered.angular.x;
    this->cmd_vel_filtered.angular.y = a * msg->angular.y + (1.0f - a) * this->cmd_vel_filtered.angular.y;
    this->cmd_vel_filtered.angular.z = a * msg->angular.z + (1.0f - a) * this->cmd_vel_filtered.angular.z;
}

void RL_Real_Go2X5::ArmJointCommandCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const std_msgs::Float32MultiArray::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const std_msgs::msg::Float32MultiArray::SharedPtr msg
#endif
)
{
    if (!msg || this->arm_command_size <= 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->arm_command_mutex);
    if (this->arm_joint_command_latest.size() != static_cast<size_t>(this->arm_command_size))
    {
        this->arm_joint_command_latest.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
    }

    const size_t count = std::min(static_cast<size_t>(this->arm_command_size), msg->data.size());
    for (size_t i = 0; i < count; ++i)
    {
        this->arm_joint_command_latest[i] = msg->data[i];
    }
}

void RL_Real_Go2X5::ArmBridgeStateCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const std_msgs::Float32MultiArray::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const std_msgs::msg::Float32MultiArray::SharedPtr msg
#endif
)
{
    if (!msg || this->arm_joint_count <= 0)
    {
        return;
    }

    const size_t n = static_cast<size_t>(this->arm_joint_count);
    std::vector<float> q(n, 0.0f);
    std::vector<float> dq(n, 0.0f);
    std::vector<float> tau(n, 0.0f);

    if (msg->data.size() >= 3 * n)
    {
        std::copy_n(msg->data.begin(), n, q.begin());
        std::copy_n(msg->data.begin() + n, n, dq.begin());
        std::copy_n(msg->data.begin() + 2 * n, n, tau.begin());
    }
    else if (msg->data.size() >= n)
    {
        std::copy_n(msg->data.begin(), n, q.begin());
    }
    else
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->arm_external_state_mutex);
    this->arm_external_state_q = std::move(q);
    this->arm_external_state_dq = std::move(dq);
    this->arm_external_state_tau = std::move(tau);
    this->arm_bridge_state_valid = true;
    this->arm_bridge_state_stamp = std::chrono::steady_clock::now();
    this->arm_bridge_state_timeout_warned = false;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandlerGo2X5(int signum)
{
    (void)signum;
    ros::shutdown();
    exit(0);
}
#elif defined(USE_CMAKE) || !defined(USE_ROS)
volatile sig_atomic_t g_shutdown_requested_go2_x5 = 0;
void signalHandlerGo2X5(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", shutting down..." << std::endl;
    g_shutdown_requested_go2_x5 = 1;
}
#endif

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " networkInterface" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }

    ChannelFactory::Instance()->Init(0, argv[1]);

#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandlerGo2X5);
    ros::init(argc, argv, "rl_sar_go2_x5");
    RL_Real_Go2X5 rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real_Go2X5>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    signal(SIGINT, signalHandlerGo2X5);
    RL_Real_Go2X5 rl_sar(argc, argv);
    while (!g_shutdown_requested_go2_x5) { sleep(1); }
    std::cout << LOGGER::INFO << "Exiting..." << std::endl;
#endif

    return 0;
}
