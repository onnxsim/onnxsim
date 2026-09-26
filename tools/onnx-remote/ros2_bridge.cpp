// Optional ROS2 control-plane bridge for the dependency-free ONNX transport.
// Tensor/profile payloads stay binary UInt8 messages; ROS2 only supplies
// discovery, health, and capability services.
#include "remote_transport.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <string>
#include <vector>

using namespace std::chrono_literals;
using UInt8MultiArray = std_msgs::msg::UInt8MultiArray;
using Trigger = std_srvs::srv::Trigger;

class OnnxRemoteBridge final : public rclcpp::Node {
 public:
  OnnxRemoteBridge() : Node("onnx_remote_bridge") {
    host_ = declare_parameter<std::string>("remote_host", "127.0.0.1");
    port_ = declare_parameter<int>("remote_port", 39501);
    connect_timeout_ms_ = declare_parameter<int>("connect_timeout_ms", 2000);
    io_timeout_ms_ = declare_parameter<int>("io_timeout_ms", 0);
    result_ = create_publisher<UInt8MultiArray>("result", rclcpp::QoS(10));
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
    capabilities_ = create_service<Trigger>(
        "capabilities", [](const std::shared_ptr<Trigger::Request>,
                            std::shared_ptr<Trigger::Response> response) {
          response->success = true;
          response->message =
              R"({"schema_version":1,"transport":"onnx-remote-v3","payload":"binary-uint8","profiling":["off","summary","detailed"],"operations":["run","compile","load_compiled","run_compiled"]})";
        });
  }

 private:
  void forward(const UInt8MultiArray::SharedPtr& message) {
    onnx_remote::Request request;
    std::string error;
    if (!onnx_remote::decode_request_payload(message->data.data(),
                                              message->data.size(), request,
                                              error)) {
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
    const bool received = sent && onnx_remote::receive_response(fd, response, error);
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

  void publish_error(const std::string& error) {
    onnx_remote::Response response;
    response.error = error;
    std::vector<uint8_t> payload;
    std::string encode_error;
    if (!onnx_remote::encode_response_payload(response, payload, encode_error)) {
      RCLCPP_ERROR(get_logger(), "%s", error.c_str());
      return;
    }
    UInt8MultiArray output;
    output.data = std::move(payload);
    result_->publish(std::move(output));
  }

  std::string host_;
  int port_;
  int connect_timeout_ms_;
  int io_timeout_ms_;
  rclcpp::Publisher<UInt8MultiArray>::SharedPtr result_;
  rclcpp::Subscription<UInt8MultiArray>::SharedPtr run_;
  rclcpp::Service<Trigger>::SharedPtr health_;
  rclcpp::Service<Trigger>::SharedPtr capabilities_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OnnxRemoteBridge>());
  rclcpp::shutdown();
  return 0;
}
