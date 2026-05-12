/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"
#include <algorithm>
#include <cmath>

RL_Sim* RL_Sim::instance = nullptr;

namespace {
// Embedded arm command for Key[4] (edit here to change the target).
const std::vector<float> kArmEmbeddedCommand = {0.6f, 3.0f, 1.0f, 0.2f, 0.0f, 0.0f};

bool PolicyControlsArm(const YamlParams& params, int arm_command_size, const std::vector<int>& arm_joint_indices)
{
    if (params.Has("policy_controls_arm"))
    {
        return params.Get<bool>("policy_controls_arm");
    }

    const int num_dofs = params.Get<int>("num_of_dofs", 0);
    if (arm_command_size <= 0 || num_dofs <= 0)
    {
        return false;
    }

    std::vector<int> arm_indices = arm_joint_indices;
    if (arm_indices.empty())
    {
        if (arm_command_size > num_dofs)
        {
            return false;
        }
        const int arm_start = num_dofs - arm_command_size;
        for (int i = 0; i < arm_command_size; ++i)
        {
            arm_indices.push_back(arm_start + i);
        }
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

    for (int action_idx : action_output_indices)
    {
        if (std::find(arm_indices.begin(), arm_indices.end(), action_idx) != arm_indices.end())
        {
            return true;
        }
    }
    return false;
}
}

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

#if defined(USE_ROS1)
    if (!ros::isInitialized())
    {
        ros::init(argc, argv, "rl_sim_mujoco", ros::init_options::NoSigintHandler);
    }
#elif defined(USE_ROS2)
    if (!rclcpp::ok())
    {
        rclcpp::init(argc, argv);
        this->ros2_initialized = true;
    }
    this->ros2_node = std::make_shared<rclcpp::Node>("rl_sim_mujoco_node");
#endif

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // arm command embedded setup (default hold)
    this->arm_command_size = this->params.Get<int>("arm_command_size", 0);
    this->arm_hold_enabled = this->params.Get<bool>("arm_hold_enabled", true);
    this->arm_joint_command_topic = this->params.Get<std::string>("arm_joint_command_topic", "/arm_joint_pos_cmd");
    this->UpdateArmJointIndices();
    if (this->arm_command_size <= 0 && !this->arm_joint_indices.empty())
    {
        this->arm_command_size = static_cast<int>(this->arm_joint_indices.size());
    }
    if (this->arm_command_size > 0)
    {
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
        this->arm_hold_position.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
        if (!this->arm_joint_indices.empty())
        {
            for (int i = 0; i < this->arm_command_size && i < static_cast<int>(this->arm_joint_indices.size()); ++i)
            {
                const int idx = this->arm_joint_indices[static_cast<size_t>(i)];
                if (idx >= 0 && static_cast<size_t>(idx) < default_pos.size())
                {
                    this->arm_hold_position[static_cast<size_t>(i)] = default_pos[static_cast<size_t>(idx)];
                }
            }
        }
        else if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            for (int i = 0; i < this->arm_command_size; ++i)
            {
                this->arm_hold_position[static_cast<size_t>(i)] = default_pos[arm_start + static_cast<size_t>(i)];
            }
        }
        const auto hold_pose = this->params.Get<std::vector<float>>("arm_hold_pose");
        if (hold_pose.size() == static_cast<size_t>(this->arm_command_size))
        {
            this->arm_hold_position = hold_pose;
            std::cout << LOGGER::INFO << "Arm hold pose (config): ";
            for (size_t i = 0; i < hold_pose.size(); ++i)
            {
                if (i) std::cout << ", ";
                std::cout << hold_pose[i];
            }
            std::cout << std::endl;
        }

        this->arm_embedded_command = kArmEmbeddedCommand;
        if (this->arm_embedded_command.size() < static_cast<size_t>(this->arm_command_size))
        {
            this->arm_embedded_command.resize(static_cast<size_t>(this->arm_command_size), 0.0f);
        }
        else if (this->arm_embedded_command.size() > static_cast<size_t>(this->arm_command_size))
        {
            this->arm_embedded_command.resize(static_cast<size_t>(this->arm_command_size));
        }
    }
    this->arm_output_filter_alpha = std::clamp(this->params.Get<float>("arm_output_filter_alpha", 0.1f), 0.0f, 1.0f);

#if defined(USE_ROS1)
    if (!this->arm_joint_command_topic.empty())
    {
        this->arm_joint_command_publisher =
            this->ros_node.advertise<std_msgs::Float32MultiArray>(this->arm_joint_command_topic, 1);
        this->arm_joint_command_subscriber = this->ros_node.subscribe(
            this->arm_joint_command_topic, 1, &RL_Sim::ArmCommandCallback, this);
    }
#elif defined(USE_ROS2)
    if (!this->arm_joint_command_topic.empty() && this->ros2_node)
    {
        this->arm_joint_command_publisher =
            this->ros2_node->create_publisher<std_msgs::msg::Float32MultiArray>(
                this->arm_joint_command_topic, rclcpp::SystemDefaultsQoS());
        this->arm_joint_command_subscriber =
            this->ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
                this->arm_joint_command_topic, rclcpp::SystemDefaultsQoS(),
                std::bind(&RL_Sim::ArmCommandCallback, this, std::placeholders::_1));
    }
#endif

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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    this->SetInitialPose();
    this->InitMujocoMappings();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

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

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
#if defined(USE_ROS2)
    if (this->ros2_initialized)
    {
        rclcpp::shutdown();
    }
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        if (this->sensor_imu_quat_adr >= 0)
        {
            state->imu.quaternion[0] = mj_data->sensordata[this->sensor_imu_quat_adr + 0];
            state->imu.quaternion[1] = mj_data->sensordata[this->sensor_imu_quat_adr + 1];
            state->imu.quaternion[2] = mj_data->sensordata[this->sensor_imu_quat_adr + 2];
            state->imu.quaternion[3] = mj_data->sensordata[this->sensor_imu_quat_adr + 3];
        }
        if (this->sensor_imu_gyro_adr >= 0)
        {
            state->imu.gyroscope[0] = mj_data->sensordata[this->sensor_imu_gyro_adr + 0];
            state->imu.gyroscope[1] = mj_data->sensordata[this->sensor_imu_gyro_adr + 1];
            state->imu.gyroscope[2] = mj_data->sensordata[this->sensor_imu_gyro_adr + 2];
        }

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            if (i < static_cast<int>(this->sensor_qpos_adr.size()) && this->sensor_qpos_adr[i] >= 0)
            {
                state->motor_state.q[i] = mj_data->sensordata[this->sensor_qpos_adr[i]];
            }
            if (i < static_cast<int>(this->sensor_qvel_adr.size()) && this->sensor_qvel_adr[i] >= 0)
            {
                state->motor_state.dq[i] = mj_data->sensordata[this->sensor_qvel_adr[i]];
            }
            if (i < static_cast<int>(this->sensor_tau_adr.size()) && this->sensor_tau_adr[i] >= 0)
            {
                state->motor_state.tau_est[i] = mj_data->sensordata[this->sensor_tau_adr[i]];
            }
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        const auto torque_limits = this->params.Get<std::vector<float>>("torque_limits");
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            if (i >= static_cast<int>(this->actuator_id.size()))
            {
                continue;
            }
            const int idx = this->actuator_id[i];
            if (idx < 0)
            {
                continue;
            }
            if (i >= static_cast<int>(this->sensor_qpos_adr.size()) || i >= static_cast<int>(this->sensor_qvel_adr.size()))
            {
                continue;
            }
            if (this->sensor_qpos_adr[i] < 0 || this->sensor_qvel_adr[i] < 0)
            {
                continue;
            }
            float ctrl =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->sensor_qpos_adr[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->sensor_qvel_adr[i]]);
            if (!torque_limits.empty() && i < static_cast<int>(torque_limits.size()))
            {
                const float limit = torque_limits[static_cast<size_t>(i)];
                if (limit > 0.0f)
                {
                    ctrl = std::clamp(ctrl, -limit, limit);
                }
            }
            mj_data->ctrl[idx] = ctrl;
        }
    }
}

void RL_Sim::InitMujocoMappings()
{
    if (!this->mj_model)
    {
        return;
    }

    const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    this->sensor_qpos_adr.assign(num_dofs, -1);
    this->sensor_qvel_adr.assign(num_dofs, -1);
    this->sensor_tau_adr.assign(num_dofs, -1);
    this->actuator_id.assign(num_dofs, -1);

    std::vector<int> joint_ids(num_dofs, -1);
    for (int i = 0; i < num_dofs && i < static_cast<int>(joint_names.size()); ++i)
    {
        joint_ids[i] = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (joint_ids[i] < 0)
        {
            std::cout << LOGGER::WARNING << "Joint not found in MJCF: " << joint_names[i] << std::endl;
        }
    }

    // Sensors
    for (int s = 0; s < this->mj_model->nsensor; ++s)
    {
        const int type = this->mj_model->sensor_type[s];
        const int objid = this->mj_model->sensor_objid[s];
        const int adr = this->mj_model->sensor_adr[s];
        for (int i = 0; i < num_dofs; ++i)
        {
            if (joint_ids[i] >= 0 && objid == joint_ids[i])
            {
                if (type == mjSENS_JOINTPOS)
                {
                    this->sensor_qpos_adr[i] = adr;
                }
                else if (type == mjSENS_JOINTVEL)
                {
                    this->sensor_qvel_adr[i] = adr;
                }
                else if (type == mjSENS_JOINTACTFRC)
                {
                    this->sensor_tau_adr[i] = adr;
                }
            }
        }
    }

    // Actuators
    for (int a = 0; a < this->mj_model->nu; ++a)
    {
        const int jid = this->mj_model->actuator_trnid[2 * a];
        for (int i = 0; i < num_dofs; ++i)
        {
            if (joint_ids[i] >= 0 && jid == joint_ids[i])
            {
                this->actuator_id[i] = a;
                break;
            }
        }
    }

    // IMU sensors by name
    {
        const int sid = mj_name2id(this->mj_model, mjOBJ_SENSOR, "imu_quat");
        if (sid >= 0)
        {
            this->sensor_imu_quat_adr = this->mj_model->sensor_adr[sid];
        }
    }
    {
        const int sid = mj_name2id(this->mj_model, mjOBJ_SENSOR, "imu_gyro");
        if (sid >= 0)
        {
            this->sensor_imu_gyro_adr = this->mj_model->sensor_adr[sid];
        }
    }
    {
        const int sid = mj_name2id(this->mj_model, mjOBJ_SENSOR, "frame_vel");
        if (sid >= 0)
        {
            this->sensor_frame_vel_adr = this->mj_model->sensor_adr[sid];
        }
    }

    // Sanity warnings
    for (int i = 0; i < num_dofs; ++i)
    {
        if (this->sensor_qpos_adr[i] < 0 || this->sensor_qvel_adr[i] < 0 || this->sensor_tau_adr[i] < 0)
        {
            std::cout << LOGGER::WARNING << "Sensor mapping incomplete for joint: "
                      << joint_names[i] << std::endl;
        }
        if (this->actuator_id[i] < 0)
        {
            std::cout << LOGGER::WARNING << "Actuator mapping missing for joint: "
                      << joint_names[i] << std::endl;
        }
    }
}

void RL_Sim::UpdateArmJointIndices()
{
    const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    if (joint_names.empty())
    {
        return;
    }

    std::vector<int> indices;
    const std::vector<std::string> arm_names = {
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"
    };
    indices.reserve(arm_names.size());
    for (const auto& name : arm_names)
    {
        auto it = std::find(joint_names.begin(), joint_names.end(), name);
        if (it != joint_names.end())
        {
            indices.push_back(static_cast<int>(std::distance(joint_names.begin(), it)));
        }
    }

    if (!indices.empty())
    {
        std::lock_guard<std::mutex> lock(this->arm_command_mutex);
        this->arm_joint_indices = std::move(indices);
    }
}

void RL_Sim::SetInitialPose()
{
    if (!this->mj_model || !this->mj_data)
    {
        return;
    }

    // Base pose: raise a bit to avoid initial ground penetration.
    const float base_height = this->params.Get<float>("init_base_height", 0.5f);
    this->mj_data->qpos[0] = 0.0;
    this->mj_data->qpos[1] = 0.0;
    this->mj_data->qpos[2] = base_height;
    this->mj_data->qpos[3] = 1.0;
    this->mj_data->qpos[4] = 0.0;
    this->mj_data->qpos[5] = 0.0;
    this->mj_data->qpos[6] = 0.0;

    const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    std::vector<float> target_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    const bool arm_lock = this->params.Get<bool>("arm_lock", false);
    const int arm_size = this->params.Get<int>("arm_command_size", 0);
    if (arm_lock && arm_size > 0 && static_cast<int>(target_pos.size()) >= arm_size)
    {
        const auto lock_pose = this->params.Get<std::vector<float>>("arm_lock_pose");
        if (lock_pose.size() == static_cast<size_t>(arm_size))
        {
            const size_t arm_start = target_pos.size() - static_cast<size_t>(arm_size);
            for (int i = 0; i < arm_size; ++i)
            {
                target_pos[arm_start + static_cast<size_t>(i)] = lock_pose[static_cast<size_t>(i)];
            }
        }
    }
    const size_t count = std::min(joint_names.size(), target_pos.size());
    for (size_t i = 0; i < count; ++i)
    {
        int jid = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (jid >= 0)
        {
            int qadr = this->mj_model->jnt_qposadr[jid];
            this->mj_data->qpos[qadr] = target_pos[i];
        }
    }

    mj_forward(this->mj_model, this->mj_data);
}

void RL_Sim::ResetSimRuntimeState()
{
    this->arm_use_embedded_command.store(false);
    this->arm_output_pos_filtered.assign(static_cast<size_t>(std::max(0, this->arm_command_size)), 0.0f);
    this->arm_output_pos_initialized = false;
    this->arm_hold_pose_applied = false;
    this->arm_pose_active = false;
    this->base_lock_active = false;
    this->base_lock_qpos.fill(0.0);
    this->RefreshArmParamsFromConfig();
}

void RL_Sim::ValidateMujocoActuatorRanges() const
{
    if (!this->mj_model)
    {
        return;
    }

    const auto torque_limits = this->params.Get<std::vector<float>>("torque_limits");
    const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    const int num_dofs = std::min(this->params.Get<int>("num_of_dofs"), static_cast<int>(torque_limits.size()));
    for (int i = 0; i < num_dofs; ++i)
    {
        if (i >= static_cast<int>(this->actuator_id.size()))
        {
            continue;
        }
        const int actuator = this->actuator_id[static_cast<size_t>(i)];
        if (actuator < 0)
        {
            continue;
        }

        const float yaml_limit = torque_limits[static_cast<size_t>(i)];
        if (yaml_limit <= 0.0f)
        {
            continue;
        }

        if (this->mj_model->actuator_ctrllimited[actuator])
        {
            const double lo = this->mj_model->actuator_ctrlrange[2 * actuator];
            const double hi = this->mj_model->actuator_ctrlrange[2 * actuator + 1];
            const float ctrl_limit = static_cast<float>(std::max(std::abs(lo), std::abs(hi)));
            if (yaml_limit > ctrl_limit + 1e-5f)
            {
                const std::string joint_label =
                    i < static_cast<int>(joint_names.size()) ? joint_names[static_cast<size_t>(i)] : std::to_string(i);
                std::cout << LOGGER::WARNING
                          << "[MuJoCo] torque_limits[" << joint_label << "]=" << yaml_limit
                          << " exceeds actuator ctrlrange +/-" << ctrl_limit << std::endl;
            }
        }
    }
}

void RL_Sim::ValidateLoadedPolicyConfig()
{
    RL::ValidateLoadedPolicyConfig();

    const auto observations = this->params.Get<std::vector<std::string>>("observations");
    if (std::find(observations.begin(), observations.end(), "height_scan") != observations.end())
    {
        std::cout << LOGGER::INFO
                  << "[MuJoCo] height_scan observation enabled; using zero-filled flat placeholder of size "
                  << this->params.Get<int>("height_scan_size", 0) << std::endl;
    }

    this->ValidateMujocoActuatorRanges();
}

void RL_Sim::ResetEpisode()
{
    if (this->mj_model && this->mj_data)
    {
        mj_resetData(this->mj_model, this->mj_data);
        this->SetInitialPose();
        mj_forward(this->mj_model, this->mj_data);
    }

    this->ResetEpisodeState();
    this->fsm.ResetToInitialState();
    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->motiontime = 0;
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

#if defined(USE_ROS1)
    if (ros::ok())
    {
        ros::spinOnce();
    }
#elif defined(USE_ROS2)
    if (this->ros2_node)
    {
        rclcpp::spin_some(this->ros2_node);
    }
#endif

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->robot_name == "go2_x5" &&
        this->control.current_keyboard == Input::Keyboard::Num1 &&
        this->control.last_keyboard != Input::Keyboard::Num1)
    {
        const auto fixed_cmd_arm_joint_command =
            this->params.Get<std::vector<float>>("fixed_cmd_arm_joint_command", std::vector<float>{});
        if (fixed_cmd_arm_joint_command.size() == static_cast<size_t>(this->arm_command_size))
        {
            this->ApplyArmHoldPose(fixed_cmd_arm_joint_command, "Key[1] pressed: reset arm command");
        }
        this->arm_pose_active = false;
        this->base_lock_active = false;
    }

    if (this->rl_init_done)
    {
        this->RefreshArmParamsFromConfig();
    }

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        this->ResetEpisode();
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }
    auto trigger_arm_pose = [&](const std::vector<float>& pose, const char* reason)
    {
        if (!pose.empty())
        {
            this->ApplyArmHoldPose(pose, reason);
            this->PublishArmCommand(pose, reason);
        }
    };

    if (this->control.current_keyboard == Input::Keyboard::Num2 &&
        this->control.last_keyboard != Input::Keyboard::Num2)
    {
        this->RefreshArmParamsFromConfig();
        std::vector<float> pose = this->params.Get<std::vector<float>>("arm_key_pose");
        if (pose.size() < static_cast<size_t>(this->arm_command_size))
        {
            const auto seq = this->params.Get<std::vector<float>>("arm_sequence");
            if (seq.size() >= static_cast<size_t>(this->arm_command_size))
            {
                pose.assign(seq.begin(), seq.begin() + static_cast<long>(this->arm_command_size));
            }
        }
        if (pose.empty())
        {
            pose = this->params.Get<std::vector<float>>("arm_hold_pose");
        }
        trigger_arm_pose(pose, "Key[2] pressed: arm command");
        this->arm_pose_active = true;
        this->base_lock_active = false;
        this->control.last_keyboard = this->control.current_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Num3 &&
        this->control.last_keyboard != Input::Keyboard::Num3)
    {
        this->RefreshArmParamsFromConfig();
        std::vector<float> pose;
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
        if (!this->arm_joint_indices.empty())
        {
            pose.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
            for (int i = 0; i < this->arm_command_size && i < static_cast<int>(this->arm_joint_indices.size()); ++i)
            {
                const int idx = this->arm_joint_indices[static_cast<size_t>(i)];
                if (idx >= 0 && static_cast<size_t>(idx) < default_pos.size())
                {
                    pose[static_cast<size_t>(i)] = default_pos[static_cast<size_t>(idx)];
                }
            }
        }
        else if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            pose.assign(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
        }
        trigger_arm_pose(pose, "Key[3] pressed: arm restore default");
        this->arm_pose_active = false;
        this->base_lock_active = false;
        this->control.last_keyboard = this->control.current_keyboard;
    }

    if (this->arm_pose_active && this->params.Get<bool>("arm_lock_base_on_pose", true))
    {
        if (!this->base_lock_active && this->mj_data)
        {
            for (int i = 0; i < 7; ++i)
            {
                this->base_lock_qpos[static_cast<size_t>(i)] = this->mj_data->qpos[i];
            }
            this->base_lock_active = true;
        }
        if (this->base_lock_active && this->mj_data && this->mj_model)
        {
            for (int i = 0; i < 7; ++i)
            {
                this->mj_data->qpos[i] = this->base_lock_qpos[static_cast<size_t>(i)];
            }
            for (int i = 0; i < 6; ++i)
            {
                this->mj_data->qvel[i] = 0.0;
            }
            mj_forward(this->mj_model, this->mj_data);
        }
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::RefreshArmParamsFromConfig()
{
    if (!this->params.Has("arm_command_size"))
    {
        return;
    }

    this->UpdateArmJointIndices();
    int indices_size = 0;
    {
        std::lock_guard<std::mutex> lock(this->arm_command_mutex);
        indices_size = static_cast<int>(this->arm_joint_indices.size());
    }
    const int fallback_size = indices_size > 0 ? indices_size : this->arm_command_size;
    const int new_size = this->params.Get<int>("arm_command_size", fallback_size);
    if (new_size <= 0)
    {
        return;
    }

    if (new_size != this->arm_command_size)
    {
        this->arm_command_size = new_size;
        this->arm_output_pos_filtered.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
        this->arm_output_pos_initialized = false;
        this->arm_hold_pose_applied = false;
    }

    this->arm_hold_enabled = this->params.Get<bool>("arm_hold_enabled", this->arm_hold_enabled);

    if (this->arm_command_size <= 0)
    {
        return;
    }

    if (!this->arm_hold_pose_applied)
    {
        std::vector<float> pose;
        const auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
        if (!this->arm_joint_indices.empty())
        {
            pose.assign(static_cast<size_t>(this->arm_command_size), 0.0f);
            for (int i = 0; i < this->arm_command_size && i < static_cast<int>(this->arm_joint_indices.size()); ++i)
            {
                const int idx = this->arm_joint_indices[static_cast<size_t>(i)];
                if (idx >= 0 && static_cast<size_t>(idx) < default_pos.size())
                {
                    pose[static_cast<size_t>(i)] = default_pos[static_cast<size_t>(idx)];
                }
            }
        }
        else if (default_pos.size() >= static_cast<size_t>(this->arm_command_size))
        {
            const size_t arm_start = default_pos.size() - static_cast<size_t>(this->arm_command_size);
            pose.assign(default_pos.begin() + static_cast<long>(arm_start), default_pos.end());
        }

        const auto hold_pose = this->params.Get<std::vector<float>>("arm_hold_pose");
        if (hold_pose.size() == static_cast<size_t>(this->arm_command_size))
        {
            pose = hold_pose;
        }

        if (!pose.empty())
        {
            {
                std::lock_guard<std::mutex> lock(this->arm_command_mutex);
                this->arm_hold_position = pose;
            }
            this->arm_hold_pose_applied = true;
        }
    }
}

void RL_Sim::ApplyArmHoldPose(const std::vector<float>& pose, const char* reason)
{
    if (this->arm_command_size <= 0)
    {
        this->UpdateArmJointIndices();
        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            if (!this->arm_joint_indices.empty())
            {
                this->arm_command_size = static_cast<int>(this->arm_joint_indices.size());
            }
        }
        if (this->arm_command_size <= 0)
        {
            std::cout << LOGGER::WARNING << "Arm hold ignored: arm_command_size <= 0" << std::endl;
            return;
        }
    }
    if (pose.size() < static_cast<size_t>(this->arm_command_size))
    {
        std::cout << LOGGER::WARNING << "Arm hold ignored: target size " << pose.size()
                  << " < arm_command_size " << this->arm_command_size << std::endl;
        return;
    }
    std::vector<float> trimmed = pose;
    if (trimmed.size() > static_cast<size_t>(this->arm_command_size))
    {
        trimmed.resize(static_cast<size_t>(this->arm_command_size));
    }
    {
        std::lock_guard<std::mutex> lock(this->arm_command_mutex);
        this->arm_hold_position = trimmed;
    }
    this->arm_hold_enabled = true;
    this->arm_use_embedded_command.store(false);
    this->arm_hold_pose_applied = true;
    (void)reason;
}

void RL_Sim::PublishArmCommand(const std::vector<float>& pose, const char* reason)
{
    if (this->arm_command_size <= 0)
    {
        this->UpdateArmJointIndices();
        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            if (!this->arm_joint_indices.empty())
            {
                this->arm_command_size = static_cast<int>(this->arm_joint_indices.size());
            }
        }
        if (this->arm_command_size <= 0)
        {
            std::cout << LOGGER::WARNING << "Arm command ignored: arm_command_size <= 0" << std::endl;
            return;
        }
    }
    if (pose.size() < static_cast<size_t>(this->arm_command_size))
    {
        std::cout << LOGGER::WARNING << "Arm command ignored: target size " << pose.size()
                  << " < arm_command_size " << this->arm_command_size << std::endl;
        return;
    }
    std::vector<float> trimmed = pose;
    if (trimmed.size() > static_cast<size_t>(this->arm_command_size))
    {
        trimmed.resize(static_cast<size_t>(this->arm_command_size));
    }

#if defined(USE_ROS1)
    if (this->arm_joint_command_publisher)
    {
        std_msgs::Float32MultiArray msg;
        msg.data = trimmed;
        this->arm_joint_command_publisher.publish(msg);
    }
#elif defined(USE_ROS2)
    if (this->arm_joint_command_publisher)
    {
        std_msgs::msg::Float32MultiArray msg;
        msg.data = trimmed;
        this->arm_joint_command_publisher->publish(msg);
    }
#endif

    std::cout << std::endl << LOGGER::INFO << reason;
    if (!this->arm_joint_command_topic.empty())
    {
        std::cout << " (topic: " << this->arm_joint_command_topic << ")";
    }
    std::cout << std::endl;
}

#if defined(USE_ROS1)
void RL_Sim::ArmCommandCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    if (!msg)
    {
        return;
    }
    this->ApplyArmHoldPose(msg->data, "ROS arm command");
}
#elif defined(USE_ROS2)
void RL_Sim::ArmCommandCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }
    this->ApplyArmHoldPose(msg->data, "ROS arm command");
}
#endif

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        const std::lock_guard<std::mutex> runtime_lock(this->runtime_state_mutex);
        if (!this->rl_init_done || !simulation_running)
        {
            return;
        }

        this->RefreshArmParamsFromConfig();
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        std::vector<float> base_lin_vel_world = {0.0f, 0.0f, 0.0f};
        if (this->sensor_frame_vel_adr >= 0)
        {
            base_lin_vel_world[0] = static_cast<float>(this->mj_data->sensordata[this->sensor_frame_vel_adr + 0]);
            base_lin_vel_world[1] = static_cast<float>(this->mj_data->sensordata[this->sensor_frame_vel_adr + 1]);
            base_lin_vel_world[2] = static_cast<float>(this->mj_data->sensordata[this->sensor_frame_vel_adr + 2]);
        }
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.lin_vel = QuatRotateInverse(this->obs.base_quat, base_lin_vel_world);
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        const int arm_command_size = this->arm_command_size;
        const bool use_embedded = this->arm_use_embedded_command.load();
        std::vector<float> arm_hold_position_local;
        std::vector<int> arm_joint_indices_local;
        {
            std::lock_guard<std::mutex> lock(this->arm_command_mutex);
            arm_hold_position_local = this->arm_hold_position;
            arm_joint_indices_local = this->arm_joint_indices;
        }
        const bool policy_controls_arm = PolicyControlsArm(this->params, arm_command_size, arm_joint_indices_local);
        const bool use_arm_hold = this->arm_hold_enabled && !use_embedded &&
                                  (this->arm_pose_active || !policy_controls_arm);
        const bool arm_lock = this->params.Get<bool>("arm_lock", false);
        const std::vector<float> &arm_lock_pose = this->arm_lock_pose_runtime_valid
            ? this->arm_lock_pose_runtime
            : this->params.Get<std::vector<float>>("arm_lock_pose");
        if (!this->arm_hold_pose_applied && arm_command_size > 0)
        {
            const auto hold_pose = this->params.Get<std::vector<float>>("arm_hold_pose");
            if (hold_pose.size() == static_cast<size_t>(arm_command_size))
            {
                {
                    std::lock_guard<std::mutex> lock(this->arm_command_mutex);
                    this->arm_hold_position = hold_pose;
                    arm_hold_position_local = hold_pose;
                }
                this->arm_hold_pose_applied = true;
                std::cout << LOGGER::INFO << "Arm hold pose (config): ";
                for (size_t i = 0; i < hold_pose.size(); ++i)
                {
                    if (i) std::cout << ", ";
                    std::cout << hold_pose[i];
                }
                std::cout << std::endl;
            }
        }
        if (arm_command_size > 0)
        {
            std::vector<float> arm_command = use_embedded ? this->arm_embedded_command : arm_hold_position_local;
            if (arm_lock)
            {
                if (arm_lock_pose.size() == static_cast<size_t>(arm_command_size))
                {
                    arm_command = arm_lock_pose;
                }
            }
            this->obs.arm_joint_command = arm_command;
        }

        const std::vector<float> policy_actions = this->Forward();
        this->last_policy_actions = policy_actions;
        this->obs.actions = this->PadActionsForObservation(policy_actions);
        const std::vector<float> sim2sim_actions = this->ApplySim2SimActionTransform(policy_actions);
        this->ComputeOutput(sim2sim_actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        const bool use_arm_policy_filter = arm_command_size > 0 && policy_controls_arm && !use_embedded && !use_arm_hold;
        if ((use_embedded || use_arm_hold || use_arm_policy_filter) && arm_command_size > 0)
        {
            if (this->arm_output_pos_filtered.size() != static_cast<size_t>(arm_command_size))
            {
                this->arm_output_pos_filtered.assign(static_cast<size_t>(arm_command_size), 0.0f);
                this->arm_output_pos_initialized = false;
            }

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

            std::vector<float> arm_hold_target = arm_hold_position_local;
            if (arm_lock)
            {
                if (arm_lock_pose.size() == static_cast<size_t>(arm_command_size))
                {
                    arm_hold_target = arm_lock_pose;
                }
            }

            for (int i = 0; i < arm_command_size; ++i)
            {
                int idx = -1;
                if (!arm_joint_indices_local.empty() && i < static_cast<int>(arm_joint_indices_local.size()))
                {
                    idx = arm_joint_indices_local[static_cast<size_t>(i)];
                }
                else
                {
                    const int num_dofs = this->params.Get<int>("num_of_dofs");
                    if (num_dofs >= arm_command_size)
                    {
                        idx = num_dofs - arm_command_size + i;
                    }
                }
                if (idx < 0 || static_cast<size_t>(idx) >= this->output_dof_pos.size())
                {
                    continue;
                }

                float target = this->output_dof_pos[static_cast<size_t>(idx)];
                if (use_arm_hold && i < static_cast<int>(arm_hold_target.size()))
                {
                    target = arm_hold_target[static_cast<size_t>(i)];
                }

                float prev = target;
                if (!this->arm_output_pos_initialized)
                {
                    if (static_cast<size_t>(idx) < this->obs.dof_pos.size())
                    {
                        prev = this->obs.dof_pos[static_cast<size_t>(idx)];
                    }
                }
                else
                {
                    prev = this->arm_output_pos_filtered[static_cast<size_t>(i)];
                }

                float filtered = target;
                if (filter_alpha < 1.0f)
                {
                    filtered = (1.0f - filter_alpha) * prev + filter_alpha * target;
                }
                if (max_delta > 0.0f)
                {
                    filtered = std::clamp(filtered, prev - max_delta, prev + max_delta);
                }
                this->arm_output_pos_filtered[static_cast<size_t>(i)] = filtered;

                this->output_dof_pos[static_cast<size_t>(idx)] = this->arm_output_pos_filtered[static_cast<size_t>(i)];
                if (use_arm_hold && static_cast<size_t>(idx) < this->output_dof_vel.size())
                {
                    this->output_dof_vel[static_cast<size_t>(idx)] = 0.0f;
                }
                if (static_cast<size_t>(idx) < this->output_dof_tau.size())
                {
                    this->output_dof_tau[static_cast<size_t>(idx)] = 0.0f;
                }
            }

            this->arm_output_pos_initialized = true;
        }

        if (arm_lock && arm_command_size > 0 && (this->motiontime % 200 == 0))
        {
            std::cout << std::endl << LOGGER::INFO << "Arm lock pose: ";
            for (int i = 0; i < arm_command_size; ++i)
            {
                if (i) std::cout << ", ";
                if (static_cast<size_t>(i) < arm_lock_pose.size())
                {
                    std::cout << arm_lock_pose[static_cast<size_t>(i)];
                }
                else
                {
                    std::cout << 0.0f;
                }
            }
            std::cout << std::endl;
        }

        auto build_arm_tracking_gains = [&](std::vector<float>& kp_out, std::vector<float>& kd_out)
        {
            if (!(use_arm_hold && arm_command_size > 0))
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
            const bool has_arm_track_kp = arm_track_kp.size() == static_cast<size_t>(arm_command_size);
            const bool has_arm_track_kd = arm_track_kd.size() == static_cast<size_t>(arm_command_size);

            for (int i = 0; i < arm_command_size; ++i)
            {
                int idx = -1;
                if (!arm_joint_indices_local.empty() && i < static_cast<int>(arm_joint_indices_local.size()))
                {
                    idx = arm_joint_indices_local[static_cast<size_t>(i)];
                }
                else
                {
                    const int num_dofs = this->params.Get<int>("num_of_dofs");
                    if (num_dofs >= arm_command_size)
                    {
                        idx = num_dofs - arm_command_size + i;
                    }
                }
                if (idx < 0 || idx >= static_cast<int>(kp_out.size()) || idx >= static_cast<int>(kd_out.size()))
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

        // Clamp final joint targets to MuJoCo joint limits to avoid unstable impulses.
        if (this->mj_model && !this->output_dof_pos.empty())
        {
            const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
            const size_t count = std::min(joint_names.size(), this->output_dof_pos.size());
            bool clamped = false;
            std::vector<std::string> clamped_joints;
            for (size_t i = 0; i < count; ++i)
            {
                const int jid = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
                if (jid < 0)
                {
                    continue;
                }
                if (this->mj_model->jnt_limited[jid])
                {
                    const double lo = this->mj_model->jnt_range[2 * jid];
                    const double hi = this->mj_model->jnt_range[2 * jid + 1];
                    float &q = this->output_dof_pos[i];
                    if (q < static_cast<float>(lo))
                    {
                        q = static_cast<float>(lo);
                        clamped = true;
                        clamped_joints.push_back(joint_names[i]);
                    }
                    else if (q > static_cast<float>(hi))
                    {
                        q = static_cast<float>(hi);
                        clamped = true;
                        clamped_joints.push_back(joint_names[i]);
                    }
                }
            }
            if (clamped && (this->motiontime % 200 == 0))
            {
                std::cout << std::endl << LOGGER::WARNING << "Clamped joint targets to limits";
                if (!clamped_joints.empty())
                {
                    std::cout << " (";
                    for (size_t i = 0; i < clamped_joints.size(); ++i)
                    {
                        if (i) std::cout << ", ";
                        std::cout << clamped_joints[i];
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
            }
        }

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
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
