#ifndef RL_REAL_GO2_X5_HPP
#define RL_REAL_GO2_X5_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_go2_x5.hpp"

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <csignal>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32MultiArray.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// union for joystick keys
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

class RL_Real_Go2X5 : public RL
{
public:
    RL_Real_Go2X5(int argc, char **argv);
    ~RL_Real_Go2X5();

#if defined(USE_ROS2) && defined(USE_ROS)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

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

    // unitree interface
    void InitLowCmd();
    int QueryMotionStatus();
    std::string QueryServiceName(std::string form, std::string name);
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateMessageHandler(const void *messages);
    void JoystickHandler(const void *message);
    MotionSwitcherClient msc;
    unitree_go::msg::dds_::LowCmd_ unitree_low_command{};
    unitree_go::msg::dds_::LowState_ unitree_low_state{};
    unitree_go::msg::dds_::WirelessController_ joystick{};
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
    xKeySwitchUnion unitree_joy;

    // arm command helpers
    void InitializeArmCommandState();
    void InitializeArmChannelConfig();
    void SetupArmCommandSubscriber();
    void SetupArmBridgeInterface();
    void ValidateJointMappingOrThrow(const char* stage) const;
    bool IsArmBridgeStateFreshLocked() const;
    bool IsArmJointIndex(int idx) const;
    void ReadArmStateFromExternal(RobotState<float> *state);
    void WriteArmCommandToExternal(const RobotCommand<float> *command);
    void ApplyArmHold(const std::vector<float>& target, const char* reason);
    bool ArmCommandDifferent(const std::vector<float>& a, const std::vector<float>& b) const;

    // others
    std::vector<float> mapped_joint_positions;
    std::vector<float> mapped_joint_velocities;

    float cmd_vel_alpha = 0.2f;
    bool cmd_vel_has_filtered = false;

    int arm_command_size = 0;
    bool arm_hold_enabled = true;
    bool arm_runtime_params_ready = false;
    std::string arm_joint_command_topic = "/arm_joint_pos_cmd";
    std::string arm_joint_command_topic_active;
    std::vector<float> arm_joint_command_latest;
    std::vector<float> arm_hold_position;
    std::vector<float> arm_command_smoothing_start;
    std::vector<float> arm_command_smoothing_target;
    std::vector<float> arm_command_smoothed;
    bool arm_command_initialized = false;
    int arm_command_smoothing_ticks = 0;
    int arm_command_smoothing_counter = 0;
    std::string arm_control_mode = "unitree";
    int arm_joint_start_index = 12;
    int arm_joint_count = 6;
    bool arm_split_control_enabled = false;
    bool arm_bridge_state_valid = false;
    bool arm_bridge_require_state = true;
    bool arm_bridge_state_timeout_warned = false;
    float arm_bridge_state_timeout_sec = 0.25f;
    std::string arm_bridge_cmd_topic = "/arx_x5/joint_cmd";
    std::string arm_bridge_state_topic = "/arx_x5/joint_state";
    std::vector<float> arm_external_state_q;
    std::vector<float> arm_external_state_dq;
    std::vector<float> arm_external_state_tau;
    std::vector<float> arm_external_shadow_q;
    std::vector<float> arm_external_shadow_dq;
    std::chrono::steady_clock::time_point arm_bridge_state_stamp;

    std::mutex cmd_vel_mutex;
    std::mutex arm_command_mutex;
    std::mutex arm_external_state_mutex;

#if defined(USE_ROS1) && defined(USE_ROS)
    std::shared_ptr<ros::NodeHandle> ros1_nh;
    geometry_msgs::Twist cmd_vel;
    geometry_msgs::Twist cmd_vel_filtered;
    ros::Subscriber cmd_vel_subscriber;
    ros::Subscriber arm_joint_command_subscriber;
    ros::Publisher arm_bridge_cmd_publisher;
    ros::Subscriber arm_bridge_state_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
    void ArmJointCommandCallback(const std_msgs::Float32MultiArray::ConstPtr &msg);
    void ArmBridgeStateCallback(const std_msgs::Float32MultiArray::ConstPtr &msg);
#elif defined(USE_ROS2) && defined(USE_ROS)
    geometry_msgs::msg::Twist cmd_vel;
    geometry_msgs::msg::Twist cmd_vel_filtered;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr arm_joint_command_subscriber;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr arm_bridge_cmd_publisher;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr arm_bridge_state_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void ArmJointCommandCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void ArmBridgeStateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
#endif
};

#endif // RL_REAL_GO2_X5_HPP
