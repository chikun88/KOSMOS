#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/plugins/action/remove_passed_goals_action.hpp"
#include "nav2_util/node_utils.hpp"
#include "yaml-cpp/yaml.h"

// Also exported for offline recorded-path replay, so the replay uses exactly
// the geometry that the BT executes. Only explicitly tagged transit gates may
// call this rule. Narrow approach/departure gates retain their original disk.
extern "C" bool omni_outer_gate_passed(
  double x, double y, double yaw, double gx, double gy,
  double next_y, double window)
{
  for (double v : {x, y, yaw, gx, gy, next_y, window}) {
    if (!std::isfinite(v)) {return false;}
  }
  if (window <= 0. || std::abs(next_y - gy) < .10 || std::abs(gx) < .01) {
    return false;
  }
  const double heading = gx > 0. ? M_PI : 0.;
  // Require the lane heading and the OUTSIDE of the central bucket. Crossing
  // its Y plane on the divider side must never discard an avoidance gate.
  return std::abs(std::remainder(yaw - heading, 2.*M_PI)) <= .04 &&
         (gx > 0. ? x >= gx : x <= gx) &&
         (next_y > gy ? y >= gy + .02 : y <= gy - .02) &&
         std::hypot(x - gx, y - gy) <= window;
}

namespace omni_route_bt
{
using Goals = std::vector<geometry_msgs::msg::PoseStamped>;

class RemovePassedBucketGoals : public BT::SyncActionNode
{
public:
  RemovePassedBucketGoals(const std::string & name, const BT::NodeConfiguration & config)
  : BT::SyncActionNode(name, config)
  {
    node_ = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    tf_ = config.blackboard->get<std::shared_ptr<tf2_ros::Buffer>>("tf_buffer");
    nav2_util::declare_parameter_if_not_declared(
      node_, "fixed_bucket_routes_file", rclcpp::ParameterValue(std::string("")));
    const auto file = node_->get_parameter("fixed_bucket_routes_file").as_string();
    const auto document = YAML::LoadFile(file);
    if (document["frame_id"].as<std::string>() != "map") {
      throw std::runtime_error("bucket passage requires map-frame routes");
    }
    for (const auto & point : document["fixed_bucket_transit"]["waypoints"]) {
      if (!point["outer_y_passage"] || !point["outer_y_passage"].as<bool>()) {continue;}
      const double x = point["x"].as<double>(), y = point["y"].as<double>();
      if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) < .01) {
        throw std::runtime_error("invalid bucket passage gate");
      }
      gates_.push_back({x, y});
      gates_.push_back({-x, y});
    }
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<Goals>("input_goals"), BT::OutputPort<Goals>("output_goals"),
      BT::InputPort<double>("radius", .16, "Unchanged gate capture disk"),
      BT::InputPort<double>("passage_window", .45, "Bounded outer passage region"),
      BT::InputPort<std::string>("robot_base_frame", "base_link", "Robot frame")};
  }

  BT::NodeStatus tick() override
  {
    auto goals = getInput<Goals>("input_goals");
    if (!goals) {return BT::NodeStatus::FAILURE;}
    auto remaining = goals.value();
    // Only join observations belonging to this exact remaining goal sequence.
    // A replacement/retry must prove passage again, including identical XY
    // goals with a new request stamp.
    const bool same_request = have_previous_ && remaining == previous_goals_;
    have_previous_ = false;
    if (remaining.size() <= 1) {
      setOutput("output_goals", remaining);
      return BT::NodeStatus::SUCCESS;
    }
    const double radius = getInput<double>("radius").value();
    const double window = getInput<double>("passage_window").value();
    if (!std::isfinite(radius) || radius <= 0. || !std::isfinite(window) || window <= 0.) {
      return BT::NodeStatus::FAILURE;
    }
    geometry_msgs::msg::PoseStamped pose;
    double stamp = 0.;
    // A stale/missing transform cannot prove passage. Keep the goals and let
    // the existing tracking/safety watchdogs handle the missing localization.
    try {
      const auto transform = tf_->lookupTransform(
        "map", getInput<std::string>("robot_base_frame").value(), tf2::TimePointZero);
      const double age = (node_->now() - rclcpp::Time(transform.header.stamp)).seconds();
      if (age < -.02 || age > .30) {
        setOutput("output_goals", remaining);
        return BT::NodeStatus::SUCCESS;
      }
      pose.pose.position.x = transform.transform.translation.x;
      pose.pose.position.y = transform.transform.translation.y;
      pose.pose.orientation = transform.transform.rotation;
      stamp = rclcpp::Time(transform.header.stamp).seconds();
    } catch (const tf2::TransformException &) {
      setOutput("output_goals", remaining);
      return BT::NodeStatus::SUCCESS;
    }
    const auto & p = pose.pose.position;
    const auto & q = pose.pose.orientation;
    const double yaw = std::atan2(2.*(q.w*q.z + q.x*q.y), 1.-2.*(q.y*q.y + q.z*q.z));
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(yaw)) {
      setOutput("output_goals", remaining);
      return BT::NodeStatus::SUCCESS;
    }
    const double dt = stamp - previous_stamp_;
    const double dx = p.x - previous_x_, dy = p.y - previous_y_;
    const double travel_squared = dx*dx + dy*dy;
    // At sprint speed, discrete TF observations can straddle the entire
    // 32 cm capture disk. Check the observed segment without enlarging it.
    // Do not bridge stale TF, long gaps, or implausible localization jumps.
    const bool sweep = same_request && dt > 0. && dt <= .15 &&
      travel_squared > 1.e-12 && std::sqrt(travel_squared) <= 4.4*dt + .02;
    double cursor = 0.;
    while (remaining.size() > 1) {
      const auto & goal = remaining.front();
      if (goal.header.frame_id != "map" || remaining[1].header.frame_id != "map") {break;}
      const auto & g = goal.pose.position;
      bool passed = std::hypot(p.x - g.x, p.y - g.y) <= radius;
      double passage_fraction = 1.;
      if (sweep) {
        const double fraction = std::clamp(
          ((g.x - previous_x_)*dx + (g.y - previous_y_)*dy)/travel_squared,
          cursor, 1.);
        if (std::hypot(previous_x_ + fraction*dx - g.x,
                       previous_y_ + fraction*dy - g.y) <= radius) {
          passed = true;
          passage_fraction = fraction;
        }
      }
      for (const auto & gate : gates_) {
        if (std::hypot(g.x - gate.first, g.y - gate.second) < 1.e-4) {
          passed |= omni_outer_gate_passed(
            p.x, p.y, yaw, g.x, g.y, remaining[1].pose.position.y, window);
        }
      }
      if (!passed) {break;}
      cursor = passage_fraction;
      remaining.erase(remaining.begin());
    }
    previous_x_ = p.x;
    previous_y_ = p.y;
    previous_stamp_ = stamp;
    previous_goals_ = remaining;
    have_previous_ = true;
    setOutput("output_goals", remaining);
    return BT::NodeStatus::SUCCESS;
  }

private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::vector<std::pair<double, double>> gates_;
  Goals previous_goals_;
  bool have_previous_ = false;
  double previous_x_ = 0., previous_y_ = 0., previous_stamp_ = 0.;
};
}  // namespace omni_route_bt

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<omni_route_bt::RemovePassedBucketGoals>("RemovePassedBucketGoals");
}
