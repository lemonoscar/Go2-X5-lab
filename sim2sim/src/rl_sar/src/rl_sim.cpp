/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace {
bool PolicyControlsArm(const YamlParams& params, int arm_command_size)
{
    if (params.Has("policy_controls_arm"))
    {
        return params.Get<bool>("policy_controls_arm");
    }

    const int num_dofs = params.Get<int>("num_of_dofs", 0);
    if (arm_command_size <= 0 || num_dofs <= 0 || arm_command_size > num_dofs)
    {
        return false;
    }

    const int num_actions = params.Get<int>("num_actions", num_dofs);
    std::vector<int> action_output_indices = params.Get<std::vector<int>>("action_output_indices");
    if (action_output_indices.empty())
    {
        action_output_indices.reserve(static_cast<size_t>(std::max(0, num_actions)));
        for (int i = 0; i < num_actions; ++i)
        {
            action_output_indices.push_back(i);
        }
    }

    const int arm_start = num_dofs - arm_command_size;
    for (int idx : action_output_indices)
    {
        if (idx >= arm_start && idx < num_dofs)
        {
            return true;
        }
    }
    return false;
}
}

RL_Sim::RL_Sim(int argc, char **argv)
{
#if defined(USE_ROS1)
    this->ang_vel_axis = "world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)
    ros2_node = std::make_shared<rclcpp::Node>("rl_sim_node");
    this->ang_vel_axis = "body";
    this->ros_namespace = ros2_node->get_namespace();
    // get params from param_node
    param_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            throw std::runtime_error("param_node service wait interrupted");
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(ros2_node->get_node_base_interface(), future, std::chrono::seconds(5));
    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
            throw std::runtime_error("param_node response missing parameters");
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
            std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
            std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
        throw std::runtime_error("param_node service call failed");
    }
#endif

    // read params from yaml
    if (this->robot_name.empty())
    {
        throw std::runtime_error("robot_name is empty; cannot load config");
    }
    this->ReadYaml(this->robot_name, "base.yaml");
    this->cmd_vel_alpha = std::clamp(this->params.Get<float>("cmd_vel_alpha", 0.2f), 0.0f, 1.0f);

    // init optional observation sources (height scan + arm command)
    this->height_scan_width = this->params.Get<int>("height_scan_width", 17);
    this->height_scan_height = this->params.Get<int>("height_scan_height", 11);
    this->height_scan_offset = this->params.Get<float>("height_scan_offset", 0.5f);
    this->height_scan_clip = this->params.Get<float>("height_scan_clip", 1.0f);
    this->height_scan_flip_x = this->params.Get<bool>("height_scan_flip_x", false);
    this->height_scan_flip_y = this->params.Get<bool>("height_scan_flip_y", false);
    this->height_scan_topic = this->params.Get<std::string>("height_scan_topic", "/height_scan/depth");
    this->arm_joint_command_topic = this->params.Get<std::string>("arm_joint_command_topic", "/arm_joint_pos_cmd");
    this->arm_command_size = this->params.Get<int>("arm_command_size", 6);
    this->arm_hold_enabled = this->params.Get<bool>("arm_hold_enabled", true);
    this->arm_output_filter_alpha =
        std::clamp(this->params.Get<float>("arm_output_filter_alpha", 0.1f), 0.0f, 1.0f);

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

    // init robot
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
#endif
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    this->height_scan_latest.resize(this->params.Get<int>("height_scan_size", this->height_scan_width * this->height_scan_height), 0.0f);
    this->arm_joint_command_latest.resize(this->arm_command_size, 0.0f);
    this->arm_sequence_steps = this->params.Get<int>("arm_sequence_steps", 0);
    this->arm_sequence_flat = this->params.Get<std::vector<float>>("arm_sequence", {});
    const float seq_interval = this->params.Get<float>("arm_sequence_interval", 2.0f);
    const float smooth_time = this->params.Get<float>("arm_command_smoothing_time", 0.2f);
    const float step_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    if (step_dt > 0.0f)
    {
        this->arm_sequence_interval_ticks = std::max(1, static_cast<int>(std::lround(seq_interval / step_dt)));
        this->arm_command_smoothing_ticks = std::max(0, static_cast<int>(std::lround(smooth_time / step_dt)));
    }
    if (this->arm_sequence_steps > 0)
    {
        const size_t required = static_cast<size_t>(this->arm_sequence_steps * this->arm_command_size);
        if (this->arm_sequence_flat.size() < required)
        {
            std::cout << LOGGER::WARNING << "arm_sequence length (" << this->arm_sequence_flat.size()
                      << ") is less than steps*size (" << required << "); disable sequence." << std::endl;
            this->arm_sequence_steps = 0;
            this->arm_sequence_flat.clear();
        }
    }
    this->arm_sequence_current.assign(this->arm_command_size, 0.0f);
    this->arm_command_smoothing_start.assign(this->arm_command_size, 0.0f);
    this->arm_command_smoothing_target.assign(this->arm_command_size, 0.0f);
    this->arm_command_smoothed.assign(this->arm_command_size, 0.0f);
    this->arm_output_pos_filtered.assign(this->arm_command_size, 0.0f);
    this->arm_command_initialized = false;
    this->arm_output_pos_initialized = false;
    {
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
        if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            this->arm_hold_position.assign(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
        }
        else
        {
            this->arm_hold_position.assign(this->arm_command_size, 0.0f);
        }
    }
    if (this->arm_joint_command_latest.size() == this->arm_hold_position.size())
    {
        this->arm_joint_command_latest = this->arm_hold_position;
        this->arm_command_smoothed = this->arm_hold_position;
        this->arm_command_smoothing_start = this->arm_hold_position;
        this->arm_command_smoothing_target = this->arm_hold_position;
        this->arm_command_initialized = true;
    }

#if defined(USE_ROS1)
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    this->StartJointController(this->ros_namespace, this->params.Get<std::vector<std::string>>("joint_names"));
    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    // subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->GazeboImuCallback(msg);}
    );
    this->model_state_subscriber = ros2_node->create_subscription<gazebo_msgs::msg::ModelStates>(
        "/gazebo/model_states", rclcpp::SystemDefaultsQoS(),
        [this] (const gazebo_msgs::msg::ModelStates::SharedPtr msg) {this->ModelStatesCallback(msg);}
    );
    this->model_state_subscriber_compat = ros2_node->create_subscription<gazebo_msgs::msg::ModelStates>(
        "/model_states", rclcpp::SystemDefaultsQoS(),
        [this] (const gazebo_msgs::msg::ModelStates::SharedPtr msg) {this->ModelStatesCallback(msg);}
    );
    this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );

    if (!this->height_scan_topic.empty())
    {
        this->height_scan_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Image>(
            this->height_scan_topic, rclcpp::SystemDefaultsQoS(),
            [this] (const sensor_msgs::msg::Image::SharedPtr msg) {this->HeightScanCallback(msg);}
        );
    }

    if (!this->arm_joint_command_topic.empty())
    {
        this->arm_joint_command_subscriber = ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
            this->arm_joint_command_topic, rclcpp::SystemDefaultsQoS(),
            [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->ArmJointCommandCallback(msg);}
        );
    }

    // service
    this->gazebo_pause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = ros2_node->create_client<std_srvs::srv::Empty>("/reset_world");

    if (this->gazebo_reset_world_client->wait_for_service(std::chrono::seconds(5)))
    {
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        this->gazebo_reset_world_client->async_send_request(empty_request);
    }
    else
    {
        std::cout << LOGGER::WARNING << "Gazebo /reset_world service not available; skip initial reset." << std::endl;
    }
#endif

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::ValidateLoadedPolicyConfig()
{
    RL::ValidateLoadedPolicyConfig();
}

void RL_Sim::ResetSimRuntimeState()
{
    {
        std::lock_guard<std::mutex> lock(this->height_scan_mutex);
        std::fill(this->height_scan_latest.begin(), this->height_scan_latest.end(), 0.0f);
    }
    {
        std::lock_guard<std::mutex> lock(this->cmd_vel_mutex);
#if defined(USE_ROS1)
        this->cmd_vel = geometry_msgs::Twist();
        this->cmd_vel_filtered = geometry_msgs::Twist();
        this->vel = geometry_msgs::Twist();
        this->pose = geometry_msgs::Pose();
#elif defined(USE_ROS2)
        this->cmd_vel = geometry_msgs::msg::Twist();
        this->cmd_vel_filtered = geometry_msgs::msg::Twist();
        {
            std::lock_guard<std::mutex> state_lock(this->state_msg_mutex);
            this->gazebo_model_twist = geometry_msgs::msg::Twist();
        }
#endif
        this->cmd_vel_has_filtered = false;
    }

    this->arm_sequence_active = false;
    this->arm_sequence_loop = false;
    this->arm_sequence_ticks = 0;
    this->arm_sequence_index = 0;
    this->arm_command_smoothing_counter = 0;
    this->arm_hold_enabled = this->params.Get<bool>("arm_hold_enabled", true);
    this->arm_pose_active = false;
    this->arm_output_pos_filtered.assign(static_cast<size_t>(std::max(0, this->arm_command_size)), 0.0f);
    this->arm_output_pos_initialized = false;

    std::vector<float> hold_pose(static_cast<size_t>(std::max(0, this->arm_command_size)), 0.0f);
    const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
    {
        const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
        hold_pose.assign(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
    }
    const auto configured_hold = this->params.Get<std::vector<float>>("arm_hold_pose");
    if (configured_hold.size() == static_cast<size_t>(this->arm_command_size))
    {
        hold_pose = configured_hold;
    }

    this->arm_hold_position = hold_pose;
    this->arm_command_smoothing_start = hold_pose;
    this->arm_command_smoothing_target = hold_pose;
    this->arm_command_smoothed = hold_pose;
    this->arm_command_initialized = !hold_pose.empty();
    {
        std::lock_guard<std::mutex> lock(this->arm_command_mutex);
        this->arm_joint_command_latest = hold_pose;
    }
}

void RL_Sim::ResetEpisode()
{
#if defined(USE_ROS1)
    std_srvs::Empty empty;
    this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
    if (this->gazebo_reset_world_client && this->gazebo_reset_world_client->wait_for_service(std::chrono::seconds(1)))
    {
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        this->gazebo_reset_world_client->async_send_request(empty_request);
    }
    else
    {
        std::cout << LOGGER::WARNING << "Gazebo /reset_world service not available; reset state locally only." << std::endl;
    }
#endif

    this->ResetEpisodeState();
    this->fsm.ResetToInitialState();
    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->motiontime = 0;
}

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path()
        / ("robot_joint_controller_params_" + std::to_string(getpid()) + ".yaml");
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "-p " + tmp_path.string() + " ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            throw std::runtime_error("Failed to start joint controller");
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}

void RL_Sim::GetState(RobotState<float> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    geometry_msgs::msg::Quaternion orientation;
    geometry_msgs::msg::Vector3 angular_velocity;
    robot_msgs::msg::RobotState robot_state_msg;
    {
        std::lock_guard<std::mutex> lock(this->state_msg_mutex);
        orientation = this->gazebo_imu.orientation;
        angular_velocity = this->gazebo_imu.angular_velocity;
        robot_state_msg = this->robot_state_subscriber_msg;
    }
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
#elif defined(USE_ROS2)
        state->motor_state.q[i] = robot_state_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = robot_state_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = robot_state_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{
    {
        std::lock_guard<std::mutex> lock(this->robot_state_mutex);
        this->GetState(&this->robot_state);
        this->StateController(&this->robot_state, &this->robot_command);
    }

    auto apply_arm_hold = [&](const std::vector<float>& target, const char* reason)
    {
        const float smooth_time = this->params.Get<float>("arm_command_smoothing_time", 0.2f);
        const float step_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
        if (step_dt > 0.0f)
        {
            this->arm_command_smoothing_ticks = std::max(0, static_cast<int>(std::lround(smooth_time / step_dt)));
        }

        if (this->arm_command_size <= 0)
        {
            std::cout << LOGGER::WARNING << "Arm hold ignored: arm_command_size <= 0" << std::endl;
            return;
        }
        if (target.size() < static_cast<size_t>(this->arm_command_size))
        {
            std::cout << LOGGER::WARNING << "Arm hold ignored: target size " << target.size()
                      << " < arm_command_size " << this->arm_command_size << std::endl;
            return;
        }

        std::vector<float> pose = target;
        if (pose.size() > static_cast<size_t>(this->arm_command_size))
        {
            pose.resize(static_cast<size_t>(this->arm_command_size));
        }

        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            this->arm_joint_command_latest = pose;
        }

        this->arm_hold_position = pose;
        this->arm_hold_enabled = true;
        this->arm_sequence_active = false;
        this->arm_sequence_loop = false;
        this->arm_sequence_ticks = 0;
        this->arm_sequence_index = 0;

        if (this->arm_command_smoothed.size() != pose.size())
        {
            this->arm_command_smoothed = pose;
            this->arm_command_smoothing_start = pose;
            this->arm_command_smoothing_target = pose;
        }
        else
        {
            this->arm_command_smoothing_start = this->arm_command_smoothed;
            this->arm_command_smoothing_target = pose;
        }

        if (this->arm_command_smoothing_ticks <= 1)
        {
            this->arm_command_smoothed = pose;
            this->arm_command_smoothing_start = pose;
        }
        this->arm_command_smoothing_counter = 0;
        this->arm_command_initialized = true;

        std::cout << LOGGER::INFO << reason << " (smooth=" << smooth_time << "s)" << std::endl;
    };

    if (this->robot_name == "go2_x5" &&
        this->control.current_keyboard == Input::Keyboard::Num1 &&
        this->control.last_keyboard != Input::Keyboard::Num1)
    {
        const auto fixed_cmd_arm_joint_command =
            this->params.Get<std::vector<float>>("fixed_cmd_arm_joint_command", std::vector<float>{});
        if (fixed_cmd_arm_joint_command.size() == static_cast<size_t>(this->arm_command_size))
        {
            apply_arm_hold(fixed_cmd_arm_joint_command, "Key[1] pressed: reset arm command");
        }
        this->arm_pose_active = false;
    }

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        this->ResetEpisode();
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Num2)
    {
        if (this->robot_name == "go2_x5")
        {
            std::vector<float> pose = this->params.Get<std::vector<float>>("arm_key_pose");
            if (pose.size() < static_cast<size_t>(this->arm_command_size))
            {
                const auto seq = this->params.Get<std::vector<float>>("arm_sequence");
                if (seq.size() >= static_cast<size_t>(this->arm_command_size))
                {
                    pose.assign(seq.begin(), seq.begin() + static_cast<long>(this->arm_command_size));
                }
            }
            if (!pose.empty())
            {
                apply_arm_hold(pose, "Key[2] pressed: arm hold pose");
                this->arm_pose_active = true;
            }
        }
        else
        {
            std::cout << LOGGER::INFO << "Key[2] pressed: trigger arm sequence (once)" << std::endl;
            this->StartArmSequence(false);
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Num3)
    {
        if (this->robot_name == "go2_x5")
        {
            const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
            if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
            {
                const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
                std::vector<float> pose(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
                apply_arm_hold(pose, "Key[3] pressed: arm restore default");
                this->arm_pose_active = false;
            }
            else
            {
                std::cout << LOGGER::WARNING << "Key[3] pressed: default_dof_pos too short for arm restore" << std::endl;
            }
        }
        else
        {
            std::cout << LOGGER::INFO << "Key[3] pressed: stop arm sequence" << std::endl;
            this->StopArmSequence();
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Num4)
    {
        this->arm_hold_enabled = !this->arm_hold_enabled;
        std::cout << LOGGER::INFO << "Key[4] pressed: arm hold " << (this->arm_hold_enabled ? "ON" : "OFF") << std::endl;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    if (!msg)
    {
        return;
    }

    int model_index = -1;
    for (size_t i = 0; i < msg->name.size(); ++i)
    {
        if ((!this->gazebo_model_name.empty() && msg->name[i] == this->gazebo_model_name) ||
            (!this->robot_name.empty() && msg->name[i] == this->robot_name))
        {
            model_index = static_cast<int>(i);
            break;
        }
    }
    if (model_index < 0 && msg->name.size() == msg->twist.size() && msg->name.size() == msg->pose.size())
    {
        for (size_t i = 0; i < msg->name.size(); ++i)
        {
            if ((!this->gazebo_model_name.empty() && msg->name[i].find(this->gazebo_model_name) != std::string::npos) ||
                (!this->robot_name.empty() && msg->name[i].find(this->robot_name) != std::string::npos))
            {
                model_index = static_cast<int>(i);
                break;
            }
        }
    }
    if (model_index < 0 || static_cast<size_t>(model_index) >= msg->twist.size() ||
        static_cast<size_t>(model_index) >= msg->pose.size())
    {
        return;
    }

    this->vel = msg->twist[static_cast<size_t>(model_index)];
    this->pose = msg->pose[static_cast<size_t>(model_index)];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->state_msg_mutex);
    this->gazebo_imu = *msg;
}

void RL_Sim::ModelStatesCallback(const gazebo_msgs::msg::ModelStates::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    int model_index = -1;
    for (size_t i = 0; i < msg->name.size(); ++i)
    {
        if ((!this->gazebo_model_name.empty() && msg->name[i] == this->gazebo_model_name) ||
            (!this->robot_name.empty() && msg->name[i] == this->robot_name))
        {
            model_index = static_cast<int>(i);
            break;
        }
    }
    if (model_index < 0 && msg->name.size() == msg->twist.size())
    {
        for (size_t i = 0; i < msg->name.size(); ++i)
        {
            if ((!this->gazebo_model_name.empty() && msg->name[i].find(this->gazebo_model_name) != std::string::npos) ||
                (!this->robot_name.empty() && msg->name[i].find(this->robot_name) != std::string::npos))
            {
                model_index = static_cast<int>(i);
                break;
            }
        }
    }
    if (model_index < 0 || static_cast<size_t>(model_index) >= msg->twist.size())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->state_msg_mutex);
    this->gazebo_model_twist = msg->twist[static_cast<size_t>(model_index)];
}

void RL_Sim::HeightScanCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    const int dest_w = (this->height_scan_width > 0) ? this->height_scan_width : static_cast<int>(msg->width);
    const int dest_h = (this->height_scan_height > 0) ? this->height_scan_height : static_cast<int>(msg->height);
    const size_t dest_size = static_cast<size_t>(dest_w * dest_h);
    if (dest_size == 0)
    {
        return;
    }

    std::vector<float> scan(dest_size, 0.0f);
    const int src_w = static_cast<int>(msg->width);
    const int src_h = static_cast<int>(msg->height);
    const int copy_w = std::min(dest_w, src_w);
    const int copy_h = std::min(dest_h, src_h);

    auto set_value = [&](int x, int y, float depth)
    {
        float height = depth - this->height_scan_offset;
        if (!std::isfinite(height))
        {
            height = 0.0f;
        }
        height = std::min(std::max(height, -this->height_scan_clip), this->height_scan_clip);
        scan[static_cast<size_t>(y * dest_w + x)] = height;
    };

    if (msg->encoding == "32FC1")
    {
        const float *data = reinterpret_cast<const float *>(msg->data.data());
        for (int y = 0; y < copy_h; ++y)
        {
            int sy = this->height_scan_flip_y ? (copy_h - 1 - y) : y;
            for (int x = 0; x < copy_w; ++x)
            {
                int sx = this->height_scan_flip_x ? (copy_w - 1 - x) : x;
                float depth = data[sy * src_w + sx];
                set_value(x, y, depth);
            }
        }
    }
    else if (msg->encoding == "16UC1")
    {
        const uint16_t *data = reinterpret_cast<const uint16_t *>(msg->data.data());
        for (int y = 0; y < copy_h; ++y)
        {
            int sy = this->height_scan_flip_y ? (copy_h - 1 - y) : y;
            for (int x = 0; x < copy_w; ++x)
            {
                int sx = this->height_scan_flip_x ? (copy_w - 1 - x) : x;
                float depth = static_cast<float>(data[sy * src_w + sx]) / 1000.0f;
                set_value(x, y, depth);
            }
        }
    }
    else
    {
        std::cout << LOGGER::WARNING << "Unsupported depth encoding: " << msg->encoding << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(this->height_scan_mutex);
    this->height_scan_latest = std::move(scan);
}

void RL_Sim::ArmJointCommandCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(this->arm_command_mutex);
    if (this->arm_joint_command_latest.empty())
    {
        this->arm_joint_command_latest.resize(msg->data.size(), 0.0f);
    }

    const size_t count = std::min(this->arm_joint_command_latest.size(), msg->data.size());
    for (size_t i = 0; i < count; ++i)
    {
        this->arm_joint_command_latest[i] = msg->data[i];
    }
}
#endif

void RL_Sim::StartArmSequence(bool loop)
{
    // Refresh sequence settings from config (loaded during InitRL).
    this->arm_command_size = this->params.Get<int>("arm_command_size", this->arm_command_size);
    this->arm_sequence_steps = this->params.Get<int>("arm_sequence_steps", 0);
    this->arm_sequence_flat = this->params.Get<std::vector<float>>("arm_sequence", {});
    const float seq_interval = this->params.Get<float>("arm_sequence_interval", 2.0f);
    const float smooth_time = this->params.Get<float>("arm_command_smoothing_time", 0.2f);
    const float step_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    if (step_dt > 0.0f)
    {
        this->arm_sequence_interval_ticks = std::max(1, static_cast<int>(std::lround(seq_interval / step_dt)));
        this->arm_command_smoothing_ticks = std::max(0, static_cast<int>(std::lround(smooth_time / step_dt)));
    }
    if (this->arm_sequence_steps > 0)
    {
        const size_t required = static_cast<size_t>(this->arm_sequence_steps * this->arm_command_size);
        if (this->arm_sequence_flat.size() < required)
        {
            std::cout << LOGGER::WARNING << "arm_sequence length (" << this->arm_sequence_flat.size()
                      << ") is less than steps*size (" << required << "); disable sequence." << std::endl;
            this->arm_sequence_steps = 0;
            this->arm_sequence_flat.clear();
        }
    }

    this->arm_sequence_current.assign(this->arm_command_size, 0.0f);

    if (this->arm_sequence_steps <= 0 || this->arm_sequence_flat.empty())
    {
        std::cout << LOGGER::WARNING << "arm_sequence is not configured; ignored." << std::endl;
        return;
    }

    std::cout << LOGGER::INFO << "Arm sequence start (" << (loop ? "loop" : "once")
              << "), steps=" << this->arm_sequence_steps
              << ", interval=" << seq_interval << "s"
              << ", smoothing=" << smooth_time << "s" << std::endl;
    this->arm_sequence_active = true;
    this->arm_sequence_loop = loop;
    this->arm_sequence_index = 0;
    this->arm_sequence_ticks = 0;
    this->UpdateArmSequence();
}

void RL_Sim::StopArmSequence()
{
    this->arm_sequence_active = false;
    this->arm_sequence_loop = false;
    if (this->arm_command_smoothed.size() == static_cast<size_t>(this->arm_command_size))
    {
        this->arm_hold_position = this->arm_command_smoothed;
    }
    std::cout << LOGGER::INFO << "Arm sequence stop" << std::endl;
}

void RL_Sim::UpdateArmSequence()
{
    if (!this->arm_sequence_active || this->arm_sequence_steps <= 0)
    {
        return;
    }
    if (this->arm_sequence_interval_ticks <= 0)
    {
        this->arm_sequence_interval_ticks = 1;
    }

    if (this->arm_sequence_ticks == 0)
    {
        const size_t base = this->arm_sequence_index * static_cast<size_t>(this->arm_command_size);
        for (int i = 0; i < this->arm_command_size; ++i)
        {
            this->arm_sequence_current[i] = this->arm_sequence_flat[base + static_cast<size_t>(i)];
        }
    }

    this->arm_sequence_ticks += 1;
    if (this->arm_sequence_ticks >= this->arm_sequence_interval_ticks)
    {
        this->arm_sequence_ticks = 0;
        this->arm_sequence_index += 1;
        if (this->arm_sequence_index >= static_cast<size_t>(this->arm_sequence_steps))
        {
            if (this->arm_sequence_loop)
            {
                this->arm_sequence_index = 0;
            }
            else
            {
                this->arm_sequence_active = false;
            }
        }
    }
}

bool RL_Sim::ArmCommandDifferent(const std::vector<float>& a, const std::vector<float>& b) const
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

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
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

void RL_Sim::JoyCallback(
#if defined(USE_ROS1)
    const sensor_msgs::Joy::ConstPtr &msg
#elif defined(USE_ROS2)
    const sensor_msgs::msg::Joy::SharedPtr msg
#endif
)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    auto btn = [this](size_t idx) -> bool {
        return idx < this->joy_msg.buttons.size() && this->joy_msg.buttons[idx];
    };
    auto axis = [this](size_t idx) -> float {
        return (idx < this->joy_msg.axes.size()) ? this->joy_msg.axes[idx] : 0.0f;
    };

    if (btn(0)) this->control.SetGamepad(Input::Gamepad::A);
    if (btn(1)) this->control.SetGamepad(Input::Gamepad::B);
    if (btn(2)) this->control.SetGamepad(Input::Gamepad::X);
    if (btn(3)) this->control.SetGamepad(Input::Gamepad::Y);
    if (btn(4)) this->control.SetGamepad(Input::Gamepad::LB);
    if (btn(5)) this->control.SetGamepad(Input::Gamepad::RB);
    if (btn(9)) this->control.SetGamepad(Input::Gamepad::LStick);
    if (btn(10)) this->control.SetGamepad(Input::Gamepad::RStick);
    if (axis(7) > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (axis(7) < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (axis(6) < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (axis(6) > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (btn(4) && btn(0)) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (btn(4) && btn(1)) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (btn(4) && btn(2)) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (btn(4) && btn(3)) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (btn(4) && btn(9)) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (btn(4) && btn(10)) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (btn(4) && axis(7) > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (btn(4) && axis(7) < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (btn(4) && axis(6) < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (btn(4) && axis(6) > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (btn(5) && btn(0)) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (btn(5) && btn(1)) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (btn(5) && btn(2)) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (btn(5) && btn(3)) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (btn(5) && btn(9)) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (btn(5) && btn(10)) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (btn(5) && axis(7) > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (btn(5) && axis(7) < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (btn(5) && axis(6) < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (btn(5) && axis(6) > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (btn(4) && btn(5)) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = axis(1); // LY
    this->control.y = axis(0); // LX
    this->control.yaw = axis(3); // RX
}

#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->state_msg_mutex);
    this->robot_state_subscriber_msg = *msg;
}
#endif

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        const std::lock_guard<std::mutex> runtime_lock(this->runtime_state_mutex);
        if (!this->rl_init_done || !simulation_running)
        {
            return;
        }

        this->episode_length_buf += 1;
        RobotState<float> robot_state_snapshot;
        {
            std::lock_guard<std::mutex> lock(this->robot_state_mutex);
            robot_state_snapshot = this->robot_state;
        }
        if (this->params.Has("height_scan_width"))
        {
            const int new_width = this->params.Get<int>("height_scan_width", this->height_scan_width);
            const int new_height = this->params.Get<int>("height_scan_height", this->height_scan_height);
            if (new_width > 0 && new_height > 0 &&
                (new_width != this->height_scan_width || new_height != this->height_scan_height))
            {
                this->height_scan_width = new_width;
                this->height_scan_height = new_height;
                {
                    std::lock_guard<std::mutex> lock(this->height_scan_mutex);
                    this->height_scan_latest.assign(static_cast<size_t>(new_width * new_height), 0.0f);
                }
            }
            this->height_scan_offset = this->params.Get<float>("height_scan_offset", this->height_scan_offset);
            this->height_scan_clip = this->params.Get<float>("height_scan_clip", this->height_scan_clip);
            this->height_scan_flip_x = this->params.Get<bool>("height_scan_flip_x", this->height_scan_flip_x);
            this->height_scan_flip_y = this->params.Get<bool>("height_scan_flip_y", this->height_scan_flip_y);
        }
        if (this->params.Has("arm_command_size"))
        {
            const int new_size = this->params.Get<int>("arm_command_size", static_cast<int>(this->arm_joint_command_latest.size()));
            if (new_size > 0 && static_cast<size_t>(new_size) != this->arm_joint_command_latest.size())
            {
                {
                    std::lock_guard<std::mutex> lock(this->arm_command_mutex);
                    this->arm_joint_command_latest.assign(static_cast<size_t>(new_size), 0.0f);
                }
                this->arm_command_size = new_size;
                this->arm_sequence_current.assign(static_cast<size_t>(new_size), 0.0f);
                this->arm_command_smoothing_start.assign(static_cast<size_t>(new_size), 0.0f);
                this->arm_command_smoothing_target.assign(static_cast<size_t>(new_size), 0.0f);
                this->arm_command_smoothed.assign(static_cast<size_t>(new_size), 0.0f);
                this->arm_output_pos_filtered.assign(static_cast<size_t>(new_size), 0.0f);
                this->arm_command_initialized = false;
                this->arm_output_pos_initialized = false;
                const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
                if (default_pos.size() >= static_cast<size_t>(new_size))
                {
                    const size_t arm_start = default_pos.size() - static_cast<size_t>(new_size);
                    this->arm_hold_position.assign(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
                }
                else
                {
                    this->arm_hold_position.assign(static_cast<size_t>(new_size), 0.0f);
                }
                if (this->arm_joint_command_latest.size() == this->arm_hold_position.size())
                {
                    this->arm_joint_command_latest = this->arm_hold_position;
                    this->arm_command_smoothed = this->arm_hold_position;
                    this->arm_command_smoothing_start = this->arm_hold_position;
                    this->arm_command_smoothing_target = this->arm_hold_position;
                    this->arm_command_initialized = true;
                }
            }
            this->arm_output_filter_alpha =
                std::clamp(this->params.Get<float>("arm_output_filter_alpha", this->arm_output_filter_alpha), 0.0f, 1.0f);
        }
        this->obs.ang_vel = robot_state_snapshot.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        std::vector<float> base_lin_vel_world = {0.0f, 0.0f, 0.0f};
#if defined(USE_ROS1)
        base_lin_vel_world = {
            static_cast<float>(this->vel.linear.x),
            static_cast<float>(this->vel.linear.y),
            static_cast<float>(this->vel.linear.z)
        };
#elif defined(USE_ROS2)
        {
            std::lock_guard<std::mutex> lock(this->state_msg_mutex);
            base_lin_vel_world = {
                static_cast<float>(this->gazebo_model_twist.linear.x),
                static_cast<float>(this->gazebo_model_twist.linear.y),
                static_cast<float>(this->gazebo_model_twist.linear.z)
            };
        }
#endif
        if (this->control.navigation_mode)
        {
            geometry_msgs::msg::Twist cmd;
            {
                std::lock_guard<std::mutex> lock(this->cmd_vel_mutex);
                cmd = this->cmd_vel_has_filtered ? this->cmd_vel_filtered : this->cmd_vel;
            }
            this->obs.commands = {(float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.angular.z};
        }
        this->obs.base_quat = robot_state_snapshot.imu.quaternion;
        this->obs.lin_vel = QuatRotateInverse(this->obs.base_quat, base_lin_vel_world);
        this->obs.dof_pos = robot_state_snapshot.motor_state.q;
        this->obs.dof_vel = robot_state_snapshot.motor_state.dq;
        const bool policy_controls_arm = PolicyControlsArm(this->params, this->arm_command_size);
        if (this->params.Has("height_scan_size"))
        {
            std::lock_guard<std::mutex> lock(this->height_scan_mutex);
            if (!this->height_scan_latest.empty())
            {
                this->obs.height_scan = this->height_scan_latest;
            }
        }
        if (this->params.Has("arm_command_size"))
        {
            const bool seq_active_before = this->arm_sequence_active;
            this->UpdateArmSequence();
            std::vector<float> desired_arm_command;
            if (this->arm_sequence_active)
            {
                desired_arm_command = this->arm_sequence_current;
            }
            else
            {
                std::lock_guard<std::mutex> lock(this->arm_command_mutex);
                if (!this->arm_joint_command_latest.empty())
                {
                    desired_arm_command = this->arm_joint_command_latest;
                }
            }

            if (!desired_arm_command.empty())
            {
                if (this->arm_command_smoothed.size() != desired_arm_command.size())
                {
                    this->arm_command_smoothed = desired_arm_command;
                    this->arm_command_smoothing_start = desired_arm_command;
                    this->arm_command_smoothing_target = desired_arm_command;
                    this->arm_command_smoothing_counter = this->arm_command_smoothing_ticks;
                    this->arm_command_initialized = true;
                }

                if (!this->arm_command_initialized)
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
                if (this->arm_sequence_active)
                {
                    this->arm_hold_position = this->arm_command_smoothed;
                }
                else if (seq_active_before && !this->arm_sequence_active)
                {
                    this->arm_hold_position = this->arm_command_smoothed;
                }
                else if (!policy_controls_arm)
                {
                    this->arm_hold_position = this->arm_command_smoothed;
                }
            }
        }

        const std::vector<float> policy_actions = this->Forward();
        this->last_policy_actions = policy_actions;
        this->obs.actions = this->PadActionsForObservation(policy_actions);
        const std::vector<float> sim2sim_actions = this->ApplySim2SimActionTransform(policy_actions);
        this->ComputeOutput(sim2sim_actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        const bool use_arm_hold = this->arm_hold_enabled && !this->arm_sequence_active &&
                                  (this->arm_pose_active || !policy_controls_arm);
        const bool use_arm_policy_filter = this->arm_command_size > 0 && policy_controls_arm && !use_arm_hold;
        if (!this->output_dof_pos.empty() && this->arm_command_size > 0 && (use_arm_hold || use_arm_policy_filter))
        {
            if (this->arm_output_pos_filtered.size() != static_cast<size_t>(this->arm_command_size))
            {
                this->arm_output_pos_filtered.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
                this->arm_output_pos_initialized = false;
            }

            const int num_dofs = this->params.Get<int>("num_of_dofs");
            const int arm_start = std::max(0, num_dofs - this->arm_command_size);
            const float filter_alpha = std::clamp(
                use_arm_policy_filter
                    ? this->params.Get<float>("arm_policy_output_filter_alpha", 1.0f)
                    : this->arm_output_filter_alpha,
                0.0f, 1.0f);
            const float max_delta = std::max(
                0.0f,
                use_arm_policy_filter
                    ? this->params.Get<float>("arm_policy_output_max_delta", 0.0f)
                    : this->params.Get<float>("arm_output_max_delta", 0.0f));

            for (int i = 0; i < this->arm_command_size; ++i)
            {
                const size_t idx = static_cast<size_t>(arm_start + i);
                if (idx >= this->output_dof_pos.size())
                {
                    continue;
                }

                const float target = use_arm_hold
                    ? ((i < static_cast<int>(this->arm_hold_position.size()))
                        ? this->arm_hold_position[static_cast<size_t>(i)]
                        : this->output_dof_pos[idx])
                    : this->output_dof_pos[idx];

                if (!this->arm_output_pos_initialized)
                {
                    if (use_arm_policy_filter && idx < this->obs.dof_pos.size())
                    {
                        this->arm_output_pos_filtered[static_cast<size_t>(i)] = this->obs.dof_pos[idx];
                    }
                    else
                    {
                        this->arm_output_pos_filtered[static_cast<size_t>(i)] = target;
                    }
                }

                float filtered = this->arm_output_pos_filtered[static_cast<size_t>(i)];
                filtered += filter_alpha * (target - filtered);
                if (max_delta > 0.0f)
                {
                    const float delta = std::clamp(filtered - this->arm_output_pos_filtered[static_cast<size_t>(i)], -max_delta, max_delta);
                    filtered = this->arm_output_pos_filtered[static_cast<size_t>(i)] + delta;
                }

                this->arm_output_pos_filtered[static_cast<size_t>(i)] = filtered;
                this->output_dof_pos[idx] = filtered;
                if (idx < this->output_dof_tau.size())
                {
                    this->output_dof_tau[idx] = 0.0f;
                }
            }
            this->arm_output_pos_initialized = true;
        }
        if (!this->output_dof_vel.empty() && use_arm_hold)
        {
            const int num_dofs = this->params.Get<int>("num_of_dofs");
            if (num_dofs >= this->arm_command_size)
            {
                const int arm_start = num_dofs - this->arm_command_size;
                for (int i = 0; i < this->arm_command_size; ++i)
                {
                    const size_t idx = static_cast<size_t>(arm_start + i);
                    if (idx < this->output_dof_vel.size())
                    {
                        this->output_dof_vel[idx] = 0.0f;
                    }
                }
            }
        }

        auto build_arm_tracking_gains = [&](std::vector<float>& kp_out, std::vector<float>& kd_out)
        {
            if (!(use_arm_hold && this->arm_command_size > 0))
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
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->last_policy_actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
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

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Sim rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Sim>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#endif
    return 0;
}
