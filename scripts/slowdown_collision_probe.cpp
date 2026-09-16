// Offline calls into the installed Nav2 polygon implementation. No ROS node,
// publishers, subscriptions, TF broadcasts, or hardware commands are created.
#include <iostream>
#include "nav2_collision_monitor/polygon.hpp"

class Probe : public nav2_collision_monitor::Polygon {
public:
  Probe() : Polygon({}, "offline_probe", nullptr, "base_link", tf2::durationFromSec(0.0)) {
    sources_names_ = {"scan"};
    std::size_t count;
    std::cin >> time_before_collision_ >> simulation_time_step_ >> min_points_ >> count;
    poly_.resize(count);
    for (auto & point : poly_) std::cin >> point.x >> point.y;
  }
};

int main() {
  Probe polygon;
  Probe slow_zone;
  nav2_collision_monitor::Velocity command;
  std::size_t count;
  while (std::cin >> command.x >> command.y >> command.tw >> count) {
    std::vector<nav2_collision_monitor::Point> points(count);
    for (auto & point : points) std::cin >> point.x >> point.y;
    std::cout << polygon.getCollisionTime({{"scan", points}}, command)
              << ' ' << slow_zone.getPointsInside(points) << '\n';
  }
}
