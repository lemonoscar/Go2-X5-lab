/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
template <typename T>
void DrainQueue(tbb::concurrent_queue<T>& queue)
{
    T item;
    while (queue.try_pop(item))
    {
    }
}

int GetNumActions(const YamlParams& params)
{
    return params.Get<int>("num_actions", params.Get<int>("num_of_dofs"));
}

int GetActionObservationSize(const YamlParams& params)
{
    if (params.Has("action_observation_size"))
    {
        return params.Get<int>("action_observation_size");
    }
    return params.Get<int>("actions_observation_dim", GetNumActions(params));
}

std::vector<int> GetActionOutputIndices(const YamlParams& params)
{
    const int num_actions = GetNumActions(params);
    std::vector<int> indices = params.Get<std::vector<int>>("action_output_indices");
    if (indices.empty())
    {
        indices.resize(static_cast<size_t>(num_actions));
        std::iota(indices.begin(), indices.end(), 0);
    }
    return indices;
}

uint32_t Sha256RotateRight(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32U - bits));
}

std::string Sha256File(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open policy model for sha256: " + path);
    }

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
    data.push_back(0x80U);
    while ((data.size() % 64U) != 56U)
    {
        data.push_back(0U);
    }
    for (int i = 7; i >= 0; --i)
    {
        data.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xffU));
    }

    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    uint32_t h0 = 0x6a09e667U;
    uint32_t h1 = 0xbb67ae85U;
    uint32_t h2 = 0x3c6ef372U;
    uint32_t h3 = 0xa54ff53aU;
    uint32_t h4 = 0x510e527fU;
    uint32_t h5 = 0x9b05688cU;
    uint32_t h6 = 0x1f83d9abU;
    uint32_t h7 = 0x5be0cd19U;

    for (size_t offset = 0; offset < data.size(); offset += 64U)
    {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i)
        {
            const size_t j = offset + static_cast<size_t>(i * 4);
            w[static_cast<size_t>(i)] =
                (static_cast<uint32_t>(data[j]) << 24U) |
                (static_cast<uint32_t>(data[j + 1]) << 16U) |
                (static_cast<uint32_t>(data[j + 2]) << 8U) |
                static_cast<uint32_t>(data[j + 3]);
        }
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 = Sha256RotateRight(w[static_cast<size_t>(i - 15)], 7U) ^
                                Sha256RotateRight(w[static_cast<size_t>(i - 15)], 18U) ^
                                (w[static_cast<size_t>(i - 15)] >> 3U);
            const uint32_t s1 = Sha256RotateRight(w[static_cast<size_t>(i - 2)], 17U) ^
                                Sha256RotateRight(w[static_cast<size_t>(i - 2)], 19U) ^
                                (w[static_cast<size_t>(i - 2)] >> 10U);
            w[static_cast<size_t>(i)] = w[static_cast<size_t>(i - 16)] + s0 +
                                        w[static_cast<size_t>(i - 7)] + s1;
        }

        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        uint32_t f = h5;
        uint32_t g = h6;
        uint32_t h = h7;

        for (int i = 0; i < 64; ++i)
        {
            const uint32_t s1 = Sha256RotateRight(e, 6U) ^ Sha256RotateRight(e, 11U) ^ Sha256RotateRight(e, 25U);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + ch + k[static_cast<size_t>(i)] + w[static_cast<size_t>(i)];
            const uint32_t s0 = Sha256RotateRight(a, 2U) ^ Sha256RotateRight(a, 13U) ^ Sha256RotateRight(a, 22U);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::nouppercase;
    for (uint32_t value : {h0, h1, h2, h3, h4, h5, h6, h7})
    {
        out << std::setw(8) << value;
    }
    return out.str();
}
}

void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    const bool trigger_fixed_cmd =
        (this->control.current_keyboard == Input::Keyboard::Num1 &&
         this->control.last_keyboard != Input::Keyboard::Num1) ||
        (this->control.current_gamepad == Input::Gamepad::RB_DPadUp &&
         this->control.last_gamepad != Input::Gamepad::RB_DPadUp);
    if (this->robot_name == "go2_x5" && trigger_fixed_cmd)
    {
        this->go2_x5_fixed_cmd_pending = true;
    }

    fsm.Run();

    this->motiontime++;

    if (this->robot_name == "go2_x5")
    {
        const bool in_rl = (this->fsm.current_state_ &&
                            this->fsm.current_state_->GetStateName().find("RLLocomotion") != std::string::npos);

        if (this->control.current_keyboard == Input::Keyboard::P ||
            this->control.current_keyboard == Input::Keyboard::Num0 ||
            this->control.current_keyboard == Input::Keyboard::Num9 ||
            this->control.current_keyboard == Input::Keyboard::Space ||
            this->control.current_keyboard == Input::Keyboard::Num5 ||
            this->control.current_gamepad == Input::Gamepad::A ||
            this->control.current_gamepad == Input::Gamepad::B ||
            this->control.current_gamepad == Input::Gamepad::LB_X)
        {
            this->go2_x5_fixed_cmd_pending = false;
        }

        if (in_rl && this->go2_x5_fixed_cmd_pending)
        {
            this->control.x = this->params.Get<float>("fixed_cmd_x", 0.6f);
            this->control.y = this->params.Get<float>("fixed_cmd_y", 0.0f);
            this->control.yaw = this->params.Get<float>("fixed_cmd_yaw", 0.0f);
            this->control.navigation_mode = false;
            this->go2_x5_fixed_cmd_pending = false;

            std::cout << std::endl << LOGGER::INFO << "Key[1] pressed: fixed cmd x=" << this->control.x
                      << " y=" << this->control.y << " yaw=" << this->control.yaw << std::endl;
        }
    }

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        // Allow fixed-command (Key[1]) to be re-triggered after stop.
        this->control.last_keyboard = Input::Keyboard::Space;
    }
    if (this->control.current_keyboard == Input::Keyboard::Num5)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        if (this->control.last_keyboard != Input::Keyboard::Num5)
        {
            std::cout << std::endl << LOGGER::INFO << "Key[5] pressed: cmd zero" << std::endl;
            this->control.last_keyboard = Input::Keyboard::Num5;
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "height_scan")
        {
            obs_list.push_back(this->obs.height_scan);
        }
        else if (observation == "arm_joint_command")
        {
            obs_list.push_back(this->obs.arm_joint_command);
        }
        else if (observation == "gripper_command")
        {
            obs_list.push_back(this->obs.gripper_command);
        }
        // ============= Other Observations =============
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->motion_loader)
            {
                auto joint_pos_sdk = this->motion_loader->GetJointPos();
                auto joint_vel_sdk = this->motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                auto waist_sdk_indices = this->params.Get<std::vector<int>>("waist_joint_indices");
                std::vector<float> waist_angles = {
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[0])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[1])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[2])]
                };
                std::vector<float> robot_torso_quat_w = MotionLoader::ComputeTorsoQuat(this->obs.base_quat, waist_angles);
                std::vector<float> ref_torso_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
        else if (observation == "RoboMimic_Deploy/phase")
        {
            float motion_time = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            float count = motion_time;
            float phase = count / this->motion_length;
            std::vector<float> phase_vec = {phase};
            obs_list.push_back(phase_vec);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(GetActionObservationSize(this->params), 0.0f);
    this->obs.height_scan.clear();
    this->obs.height_scan.resize(this->params.Get<int>("height_scan_size", 0), 0.0f);
    this->obs.arm_joint_command.clear();
    this->obs.arm_joint_command.resize(this->params.Get<int>("arm_command_size", 0), 0.0f);
    this->obs.gripper_command.clear();
    this->obs.gripper_command.resize(this->params.Get<int>("gripper_command_size", 0),
                                     this->params.Get<float>("gripper_command_default", 0.0f));
    this->ComputeObservation();
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
    this->control.navigation_mode = false;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");

    // init joint num first
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
    this->ResetObservationHistory();
    this->ClearOutputQueues();
    this->ValidateLoadedPolicyConfig();
    this->ResetActionRuntimeState();

    // init model
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
    this->ValidatePolicyManifest(robot_config_path, model_path);
}

std::vector<float> RL::PadActionsForObservation(const std::vector<float>& actions) const
{
    const int observation_size = GetActionObservationSize(this->params);
    std::vector<float> padded(static_cast<size_t>(std::max(0, observation_size)),
                              this->params.Get<float>("action_observation_pad_value", 0.0f));
    const size_t copy_count = std::min(padded.size(), actions.size());
    std::copy_n(actions.begin(), static_cast<long>(copy_count), padded.begin());
    return padded;
}

std::vector<float> RL::ApplySim2SimActionTransform(const std::vector<float>& policy_actions)
{
    const int num_actions = GetNumActions(this->params);
    if (policy_actions.size() != static_cast<size_t>(num_actions))
    {
        throw std::runtime_error(
            "policy action size mismatch: expected " + std::to_string(num_actions) +
            ", got " + std::to_string(policy_actions.size()));
    }

    std::vector<float> delayed = policy_actions;
    this->sim2sim_action_history.push_back(policy_actions);
    while (this->sim2sim_action_history.size() < static_cast<size_t>(this->sim2sim_action_delay_steps + 1))
    {
        this->sim2sim_action_history.push_front(policy_actions);
    }
    while (this->sim2sim_action_history.size() > static_cast<size_t>(this->sim2sim_action_delay_steps + 1))
    {
        this->sim2sim_action_history.pop_front();
    }
    if (this->sim2sim_action_delay_steps > 0 && !this->sim2sim_action_history.empty())
    {
        delayed = this->sim2sim_action_history.front();
    }

    std::vector<float> applied = delayed;
    const float hold_prob = std::clamp(this->params.Get<float>("sim2sim_action_hold_prob", 0.0f), 0.0f, 1.0f);
    if (!this->last_applied_actions.empty() && hold_prob > 0.0f)
    {
        std::bernoulli_distribution hold_dist(hold_prob);
        if (hold_dist(this->sim2sim_rng))
        {
            applied = this->last_applied_actions;
        }
    }

    const float noise_std = std::max(0.0f, this->params.Get<float>("sim2sim_action_noise_std", 0.0f));
    if (noise_std > 0.0f)
    {
        std::normal_distribution<float> noise_dist(0.0f, noise_std);
        for (float& value : applied)
        {
            value += noise_dist(this->sim2sim_rng);
        }
    }

    this->last_applied_actions = applied;
    return applied;
}

void RL::ResetActionRuntimeState()
{
    const int num_actions = GetNumActions(this->params);
    this->last_policy_actions.assign(static_cast<size_t>(std::max(0, num_actions)), 0.0f);
    this->last_applied_actions = this->last_policy_actions;
    this->sim2sim_action_history.clear();
    this->sim2sim_action_delay_steps = std::max(0, this->params.Get<int>("sim2sim_action_delay_steps", 0));
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    // Action interface checklist:
    // 1) Policy outputs joint position offsets (same units as default_dof_pos; expected radians),
    //    except wheel_indices which are treated as velocity commands.
    // 2) Controller type is effort via PD (RobotJointController/Group uses q, dq, kp, kd, tau).
    // 3) No unit conversions or sign flips here; action_scale + default_dof_pos are applied directly.
    // 4) Action clipping happens in Forward() via clip_actions_*; no extra filtering/slew here.
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const int num_actions = GetNumActions(this->params);
    const auto action_output_indices = GetActionOutputIndices(this->params);
    if (actions.size() != static_cast<size_t>(num_actions))
    {
        throw std::runtime_error(
            "ComputeOutput action size mismatch: expected " + std::to_string(num_actions) +
            ", got " + std::to_string(actions.size()));
    }
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled(static_cast<size_t>(num_dofs), 0.0f);
    std::vector<float> vel_actions_scaled(static_cast<size_t>(num_dofs), 0.0f);
    const auto wheel_indices = this->params.Get<std::vector<int>>("wheel_indices");
    for (int action_idx = 0; action_idx < num_actions; ++action_idx)
    {
        const int dof_idx = action_output_indices[static_cast<size_t>(action_idx)];
        const float scaled = actions_scaled[static_cast<size_t>(action_idx)];
        if (std::find(wheel_indices.begin(), wheel_indices.end(), dof_idx) != wheel_indices.end())
        {
            vel_actions_scaled[static_cast<size_t>(dof_idx)] = scaled;
        }
        else
        {
            pos_actions_scaled[static_cast<size_t>(dof_idx)] = scaled;
        }
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

void RL::ClearOutputQueues()
{
    DrainQueue(this->output_cmd_queue);
    DrainQueue(this->output_dof_pos_queue);
    DrainQueue(this->output_dof_vel_queue);
    DrainQueue(this->output_dof_tau_queue);
}

void RL::ResetObservationHistory()
{
    this->history_obs.clear();

    const auto observations_history = this->params.Get<std::vector<int>>("observations_history");
    if (observations_history.empty())
    {
        this->history_obs_buf = ObservationBuffer();
        return;
    }

    int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
    this->history_obs_buf = ObservationBuffer(
        1,
        this->obs_dims,
        history_length,
        this->params.Get<std::string>("observations_history_priority"));
}

void RL::ResetEpisodeState(bool reset_control)
{
    std::lock_guard<std::mutex> runtime_lock(this->runtime_state_mutex);

    this->rl_init_done = false;
    this->motiontime = 0;
    this->episode_length_buf = 0;
    this->arm_lock_pose_runtime.clear();
    this->arm_lock_pose_runtime_valid = false;
    this->go2_x5_fixed_cmd_pending = false;

    this->InitObservations();
    this->InitOutputs();
    this->ResetActionRuntimeState();
    this->ResetObservationHistory();
    this->ClearOutputQueues();

    if (reset_control)
    {
        this->control.Reset();
    }
    else
    {
        this->InitControl();
    }

    this->ResetSimRuntimeState();
}

void RL::ValidateLoadedPolicyConfig()
{
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const int num_actions = GetNumActions(this->params);
    const int action_observation_size = GetActionObservationSize(this->params);
    if (num_dofs <= 0)
    {
        throw std::runtime_error("num_of_dofs must be positive");
    }
    if (num_actions <= 0)
    {
        throw std::runtime_error("num_actions must be positive");
    }
    if (action_observation_size < num_actions)
    {
        throw std::runtime_error("action_observation_size must be >= num_actions");
    }
    if (this->params.Has("action_observation_size") && this->params.Has("actions_observation_dim") &&
        this->params.Get<int>("action_observation_size") != this->params.Get<int>("actions_observation_dim"))
    {
        throw std::runtime_error("action_observation_size and actions_observation_dim must match");
    }

    auto require_size = [&](const std::string& key, int expected)
    {
        if (!this->params.Has(key))
        {
            return;
        }
        const size_t actual = this->params.config_node[key].size();
        if (actual != static_cast<size_t>(expected))
        {
            throw std::runtime_error(
                "Config key '" + key + "' size mismatch: expected " +
                std::to_string(expected) + ", got " + std::to_string(actual));
        }
    };

    auto require_optional_size = [&](const std::string& key, int expected)
    {
        if (!this->params.Has(key))
        {
            return;
        }
        const size_t actual = this->params.config_node[key].size();
        if (actual != 0 && actual != static_cast<size_t>(expected))
        {
            throw std::runtime_error(
                "Config key '" + key + "' size mismatch: expected 0 or " +
                std::to_string(expected) + ", got " + std::to_string(actual));
        }
    };

    require_size("default_dof_pos", num_dofs);
    require_size("action_scale", num_actions);
    require_size("rl_kp", num_dofs);
    require_size("rl_kd", num_dofs);
    require_size("torque_limits", num_dofs);
    require_optional_size("clip_actions_upper", num_actions);
    require_optional_size("clip_actions_lower", num_actions);
    require_optional_size("joint_names", num_dofs);
    require_optional_size("joint_mapping", num_dofs);
    require_optional_size("action_output_indices", num_actions);

    const auto observations = this->params.Get<std::vector<std::string>>("observations");
    if (std::find(observations.begin(), observations.end(), "height_scan") != observations.end() &&
        this->params.Get<int>("height_scan_size", 0) <= 0)
    {
        throw std::runtime_error("height_scan observation requires height_scan_size > 0");
    }
    if (std::find(observations.begin(), observations.end(), "arm_joint_command") != observations.end() &&
        this->params.Get<int>("arm_command_size", 0) <= 0)
    {
        throw std::runtime_error("arm_joint_command observation requires arm_command_size > 0");
    }
    if (std::find(observations.begin(), observations.end(), "gripper_command") != observations.end() &&
        this->params.Get<int>("gripper_command_size", 0) <= 0)
    {
        throw std::runtime_error("gripper_command observation requires gripper_command_size > 0");
    }

    const auto action_output_indices = GetActionOutputIndices(this->params);
    for (int idx : action_output_indices)
    {
        if (idx < 0 || idx >= num_dofs)
        {
            throw std::runtime_error("action_output_indices contains out-of-range dof index");
        }
    }

    if (this->params.Has("num_observations"))
    {
        const int expected = this->params.Get<int>("num_observations");
        const int actual = static_cast<int>(this->ComputeObservation().size());
        if (expected != actual)
        {
            throw std::runtime_error(
                "num_observations mismatch: expected " + std::to_string(expected) +
                ", got " + std::to_string(actual));
        }
    }
}

void RL::ValidatePolicyManifest(const std::string& robot_config_path, const std::string& model_path)
{
    const std::filesystem::path manifest_path =
        std::filesystem::path(POLICY_DIR) / robot_config_path / "manifest.yaml";
    if (!std::filesystem::exists(manifest_path))
    {
        throw std::runtime_error("Policy manifest is required but missing: " + manifest_path.string());
    }

    YAML::Node manifest_root = YAML::LoadFile(manifest_path.string());
    YAML::Node manifest = manifest_root[robot_config_path];
    if (!manifest)
    {
        throw std::runtime_error("Policy manifest missing profile key: " + robot_config_path);
    }

    auto require_string = [&](const std::string& key) -> std::string
    {
        if (!manifest[key])
        {
            throw std::runtime_error("Policy manifest missing key: " + key);
        }
        return manifest[key].as<std::string>();
    };

    auto require_int = [&](const std::string& key) -> int
    {
        if (!manifest[key])
        {
            throw std::runtime_error("Policy manifest missing key: " + key);
        }
        return manifest[key].as<int>();
    };

    const std::string manifest_mode = require_string("mode");
    const std::string config_mode = this->params.Get<std::string>("policy_mode", manifest_mode);
    if (manifest_mode != config_mode)
    {
        throw std::runtime_error("Policy manifest mode mismatch: manifest=" + manifest_mode +
                                 ", config=" + config_mode);
    }

    const std::string manifest_model_name = require_string("model_name");
    const std::string config_model_name = this->params.Get<std::string>("model_name");
    if (manifest_model_name != config_model_name)
    {
        throw std::runtime_error("Policy manifest model_name mismatch: manifest=" + manifest_model_name +
                                 ", config=" + config_model_name);
    }

    const int manifest_observation_dim = require_int("observation_dim");
    const int manifest_action_dim = require_int("action_dim");
    const int manifest_action_observation_dim = require_int("action_observation_dim");
    const int config_observation_dim = this->params.Get<int>("num_observations");
    const int config_action_dim = GetNumActions(this->params);
    const int config_action_observation_dim = GetActionObservationSize(this->params);

    if (manifest_observation_dim != config_observation_dim)
    {
        throw std::runtime_error("Policy manifest observation_dim mismatch: manifest=" +
                                 std::to_string(manifest_observation_dim) + ", config=" +
                                 std::to_string(config_observation_dim));
    }
    if (manifest_action_dim != config_action_dim)
    {
        throw std::runtime_error("Policy manifest action_dim mismatch: manifest=" +
                                 std::to_string(manifest_action_dim) + ", config=" +
                                 std::to_string(config_action_dim));
    }
    if (manifest_action_observation_dim != config_action_observation_dim)
    {
        throw std::runtime_error("Policy manifest action_observation_dim mismatch: manifest=" +
                                 std::to_string(manifest_action_observation_dim) + ", config=" +
                                 std::to_string(config_action_observation_dim));
    }

    const std::string expected_sha256 = require_string("policy_sha256");
    const std::string actual_sha256 = Sha256File(model_path);
    if (expected_sha256 != actual_sha256)
    {
        throw std::runtime_error("Policy manifest sha256 mismatch: expected=" + expected_sha256 +
                                 ", actual=" + actual_sha256);
    }

    std::vector<float> dry_run_observation(static_cast<size_t>(config_observation_dim), 0.0f);
    const std::vector<float> dry_run_actions = this->model->forward({dry_run_observation});
    if (dry_run_actions.size() != static_cast<size_t>(config_action_dim))
    {
        throw std::runtime_error("Policy dry-run action_dim mismatch: expected=" +
                                 std::to_string(config_action_dim) + ", got=" +
                                 std::to_string(dry_run_actions.size()));
    }

    std::cout << LOGGER::INFO << "Policy manifest OK: profile=" << robot_config_path
              << ", id=" << require_string("policy_id")
              << ", mode=" << manifest_mode
              << ", model=" << model_path
              << ", sha256=" << actual_sha256
              << ", shape=" << config_observation_dim << "->" << config_action_dim
              << ", action_obs=" << config_action_observation_dim << std::endl;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    RLCommandOutput output;
    bool has_output = false;
    while (rl.output_cmd_queue.try_pop(output))
    {
        has_output = true;
    }
    if (has_output)
    {
        const int num_dofs = rl.params.Get<int>("num_of_dofs");
        const auto rl_kp = rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd = rl.params.Get<std::vector<float>>("rl_kd");
        const bool use_output_kp = output.kp.size() == static_cast<size_t>(num_dofs);
        const bool use_output_kd = output.kd.size() == static_cast<size_t>(num_dofs);
        for (int i = 0; i < num_dofs; ++i)
        {
            if (static_cast<size_t>(i) < output.pos.size())
            {
                fsm_command->motor_command.q[i] = output.pos[static_cast<size_t>(i)];
            }
            if (static_cast<size_t>(i) < output.vel.size())
            {
                fsm_command->motor_command.dq[i] = output.vel[static_cast<size_t>(i)];
            }
            if (i < static_cast<int>(rl_kp.size()))
            {
                fsm_command->motor_command.kp[i] =
                    use_output_kp ? output.kp[static_cast<size_t>(i)] : rl_kp[static_cast<size_t>(i)];
            }
            if (i < static_cast<int>(rl_kd.size()))
            {
                fsm_command->motor_command.kd[i] =
                    use_output_kd ? output.kd[static_cast<size_t>(i)] : rl_kd[static_cast<size_t>(i)];
            }
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    // Optional: lock arm joints to default pose (for go2_x5 stabilization / leg-only motion).
    if (rl.params.Get<bool>("arm_lock", false))
    {
        const int num_dofs = rl.params.Get<int>("num_of_dofs");
        const int arm_size = rl.params.Get<int>("arm_command_size", 0);
        if (arm_size > 0 && num_dofs >= arm_size)
        {
            const int arm_start = num_dofs - arm_size;
            const auto default_pos = rl.params.Get<std::vector<float>>("default_dof_pos");
            const auto& arm_lock_pose = rl.arm_lock_pose_runtime_valid
                ? rl.arm_lock_pose_runtime
                : rl.params.Get<std::vector<float>>("arm_lock_pose");
            const auto fixed_kp = rl.params.Get<std::vector<float>>("fixed_kp");
            const auto fixed_kd = rl.params.Get<std::vector<float>>("fixed_kd");
            for (int i = 0; i < arm_size; ++i)
            {
                const int idx = arm_start + i;
                if (idx >= num_dofs || idx >= static_cast<int>(default_pos.size()))
                {
                    continue;
                }
                if (arm_lock_pose.size() == static_cast<size_t>(arm_size))
                {
                    fsm_command->motor_command.q[idx] = arm_lock_pose[static_cast<size_t>(i)];
                }
                else
                {
                    fsm_command->motor_command.q[idx] = default_pos[static_cast<size_t>(idx)];
                }
                fsm_command->motor_command.dq[idx] = 0.0f;
                if (idx < static_cast<int>(fixed_kp.size()))
                {
                    fsm_command->motor_command.kp[idx] = fixed_kp[static_cast<size_t>(idx)];
                }
                if (idx < static_cast<int>(fixed_kd.size()))
                {
                    fsm_command->motor_command.kd[idx] = fixed_kd[static_cast<size_t>(idx)];
                }
                fsm_command->motor_command.tau[idx] = 0.0f;
            }
        }
    }
}
