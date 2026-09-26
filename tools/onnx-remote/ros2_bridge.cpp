// Optional ROS2 control-plane bridge for the dependency-free ONNX transport.
// Tensor/profile payloads stay binary UInt8 messages; ROS2 only supplies
// discovery, health, and capability services.
#include "remote_transport.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using UInt8MultiArray = std_msgs::msg::UInt8MultiArray;
using String = std_msgs::msg::String;
using Trigger = std_srvs::srv::Trigger;

class OnnxRemoteBridge final : public rclcpp::Node {
public:
  OnnxRemoteBridge() : Node("onnx_remote_bridge") {
    host_ = declare_parameter<std::string>("remote_host", "127.0.0.1");
    port_ = declare_parameter<int>("remote_port", 39501);
    configured_host_ = host_;
    configured_port_ = port_;
    connect_timeout_ms_ = declare_parameter<int>("connect_timeout_ms", 2000);
    io_timeout_ms_ = declare_parameter<int>("io_timeout_ms", 0);
    auto_discover_ = declare_parameter<bool>("auto_discover", false);
    discovery_topic_ = declare_parameter<std::string>("discovery_topic",
                                                      "onnx_remote/runners");
    discovery_target_ = declare_parameter<std::string>("discovery_target", "");
    announce_period_ms_ = declare_parameter<int>("announce_period_ms", 5000);
    discovery_timeout_ms_ = declare_parameter<int>("discovery_timeout_ms", 15000);
    advertise_host_ = declare_parameter<std::string>("advertise_host", host_);
    runner_id_ = declare_parameter<std::string>("runner_id", get_name());
    result_ = create_publisher<UInt8MultiArray>("result", rclcpp::QoS(10));
    discovery_ = create_publisher<String>(
        discovery_topic_, rclcpp::QoS(1).transient_local().reliable());
    discovery_sub_ = create_subscription<String>(
        discovery_topic_, rclcpp::QoS(10).transient_local().reliable(),
        [this](const String::SharedPtr message) { discover(message); });
    announce_timer_ = create_wall_timer(
        std::chrono::milliseconds(std::max(100, announce_period_ms_)),
        [this]() {
          expire_discovery();
          announce();
        });
    run_ = create_subscription<UInt8MultiArray>(
        "run", rclcpp::QoS(10),
        [this](const UInt8MultiArray::SharedPtr message) { forward(message); });
    health_ = create_service<Trigger>(
        "health", [this](const std::shared_ptr<Trigger::Request>,
                         std::shared_ptr<Trigger::Response> response) {
          const int fd = onnx_remote::connect_tcp_timeout(
              host_, static_cast<uint16_t>(port_), connect_timeout_ms_);
          if (fd < 0) {
            response->success = false;
            response->message = "remote worker is unreachable";
            return;
          }
          onnx_remote::set_socket_io_timeout(fd, io_timeout_ms_);
          onnx_remote::close_socket(fd);
          response->success = true;
          response->message = "remote worker is reachable";
        });
    capabilities_ = create_service<
        Trigger>("capabilities", [](const std::shared_ptr<Trigger::Request>,
                                    std::shared_ptr<Trigger::Response>
                                        response) {
      response->success = true;
      response->message =
          R"({"schema_version":1,"transport":"onnx-remote-v4","payload":"binary-uint8","profiling":["off","summary","detailed"],"operations":["run","compile","load_compiled","run_compiled"],"tensor_dtypes":["float32","float16","bfloat16","int8","uint8","int16","uint16","int32","int64","uint32","uint64","double","bool"]})";
    });
  }

private:
  static std::string json_string(const std::string &json,
                                 const std::string &key) {
    const std::string marker = "\"" + key + "\":\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos)
      return {};
    const size_t value_start = start + marker.size();
    const size_t end = json.find('"', value_start);
    return end == std::string::npos
               ? std::string{}
               : json.substr(value_start, end - value_start);
  }

  static int json_int(const std::string &json, const std::string &key) {
    const std::string marker = "\"" + key + "\":";
    const size_t start = json.find(marker);
    if (start == std::string::npos)
      return 0;
    const size_t value_start = start + marker.size();
    return std::atoi(json.c_str() + value_start);
  }

  void announce() {
    String message;
    message.data = "{\"schema_version\":1,\"runner_id\":\"" + runner_id_ +
                   "\",\"host\":\"" + advertise_host_ +
                   "\",\"port\":" + std::to_string(port_) + ",\"target\":\"" +
                   discovery_target_ +
                   "\",\"transport\":\"onnx-remote-v4\","
                   "\"profiling\":[\"off\",\"summary\",\"detailed\"]}";
    discovery_->publish(std::move(message));
  }

  void discover(const String::SharedPtr &message) {
    if (!auto_discover_)
      return;
    const std::string id = json_string(message->data, "runner_id");
    if (id.empty() || id == runner_id_)
      return;
    const std::string target = json_string(message->data, "target");
    if (!discovery_target_.empty() && target != discovery_target_)
      return;
    const std::string host = json_string(message->data, "host");
    const int port = json_int(message->data, "port");
    if (host.empty() || port <= 0 || port > 65535)
      return;
    host_ = host;
    port_ = port;
    last_discovery_ = std::chrono::steady_clock::now();
    discovered_ = true;
    RCLCPP_INFO(get_logger(), "auto-discovered runner %s at %s:%d", id.c_str(),
                host_.c_str(), port_);
  }

  void expire_discovery() {
    if (!auto_discover_ || !discovered_ || discovery_timeout_ms_ <= 0)
      return;
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_discovery_);
    if (age.count() <= discovery_timeout_ms_)
      return;
    RCLCPP_WARN(get_logger(), "discovered runner expired after %ld ms",
                static_cast<long>(age.count()));
    host_ = configured_host_;
    port_ = configured_port_;
    discovered_ = false;
  }

  void forward(const UInt8MultiArray::SharedPtr &message) {
    onnx_remote::Request request;
    std::string error;
    if (!onnx_remote::decode_request_payload(
            message->data.data(), message->data.size(), request, error)) {
      publish_error(error);
      return;
    }
    const int fd = onnx_remote::connect_tcp_timeout(
        host_, static_cast<uint16_t>(port_), connect_timeout_ms_);
    if (fd < 0) {
      publish_error("ROS2 bridge: remote worker connection failed");
      return;
    }
    onnx_remote::set_socket_io_timeout(fd, io_timeout_ms_);
    onnx_remote::Response response;
    const bool sent = onnx_remote::send_request(fd, request, error);
    const bool received =
        sent && onnx_remote::receive_response(fd, response, error);
    onnx_remote::close_socket(fd);
    if (!received) {
      publish_error(error.empty() ? "remote worker request failed" : error);
      return;
    }
    std::vector<uint8_t> payload;
    if (!onnx_remote::encode_response_payload(response, payload, error)) {
      publish_error(error);
      return;
    }
    UInt8MultiArray output;
    output.data = std::move(payload);
    result_->publish(std::move(output));
  }

  void publish_error(const std::string &error) {
    onnx_remote::Response response;
    response.error = error;
    std::vector<uint8_t> payload;
    std::string encode_error;
    if (!onnx_remote::encode_response_payload(response, payload,
                                              encode_error)) {
      RCLCPP_ERROR(get_logger(), "%s", error.c_str());
      return;
    }
    UInt8MultiArray output;
    output.data = std::move(payload);
    result_->publish(std::move(output));
  }

  std::string host_;
  int port_;
  std::string configured_host_;
  int configured_port_;
  int connect_timeout_ms_;
  int io_timeout_ms_;
  bool auto_discover_;
  int announce_period_ms_ = 5000;
  int discovery_timeout_ms_ = 15000;
  bool discovered_ = false;
  std::chrono::steady_clock::time_point last_discovery_{};
  std::string discovery_topic_;
  std::string discovery_target_;
  std::string advertise_host_;
  std::string runner_id_;
  rclcpp::Publisher<UInt8MultiArray>::SharedPtr result_;
  rclcpp::Publisher<String>::SharedPtr discovery_;
  rclcpp::Subscription<UInt8MultiArray>::SharedPtr run_;
  rclcpp::Subscription<String>::SharedPtr discovery_sub_;
  rclcpp::TimerBase::SharedPtr announce_timer_;
  rclcpp::Service<Trigger>::SharedPtr health_;
  rclcpp::Service<Trigger>::SharedPtr capabilities_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OnnxRemoteBridge>());
  rclcpp::shutdown();
  return 0;
}
