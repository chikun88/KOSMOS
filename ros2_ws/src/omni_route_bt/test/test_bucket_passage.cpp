#include <cmath>
#include <limits>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include "gtest/gtest.h"
#include "behaviortree_cpp/bt_factory.h"
#include "nav2_behavior_tree/plugins/action/remove_passed_goals_action.hpp"

extern "C" bool omni_outer_gate_passed(double, double, double, double, double, double, double);

TEST(BucketPassage, RecordedMissesPassOnBothFields)
{
  for (const double side : {1., -1.}) {
    const double yaw = side > 0. ? M_PI : 0.;
    EXPECT_TRUE(omni_outer_gate_passed(side*1.79698644, -1.39091557, yaw,
      side*1.55, -1.25, -1.55, .45));
    EXPECT_TRUE(omni_outer_gate_passed(side*1.66629263, .33963769, yaw,
      side*1.55, .15, .50, .45));
  }
}

TEST(BucketPassage, RejectsInsideBeforeWrongHeadingFarAndInvalidPose)
{
  EXPECT_FALSE(omni_outer_gate_passed(1.4, .3, M_PI, 1.55, .15, .5, .45));
  EXPECT_FALSE(omni_outer_gate_passed(1.8, .1, M_PI, 1.55, .15, .5, .45));
  EXPECT_FALSE(omni_outer_gate_passed(1.8, .3, 0., 1.55, .15, .5, .45));
  EXPECT_FALSE(omni_outer_gate_passed(2.2, .3, M_PI, 1.55, .15, .5, .45));
  EXPECT_FALSE(omni_outer_gate_passed(1.8, .3, M_PI, 1.55, .15, .15, .45));
  EXPECT_FALSE(omni_outer_gate_passed(1.8, .3, M_PI, 1.55, .15, .5, -1.));
  EXPECT_FALSE(omni_outer_gate_passed(std::numeric_limits<double>::quiet_NaN(),
    .3, M_PI, 1.55, .15, .5, .45));
}

class BucketTree : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
  void SetUp() override
  {
    file_ = (std::filesystem::temp_directory_path() /
      ("bucket-routes-test-" + std::to_string(getpid()) + ".yaml")).string();
    std::ofstream(file_) << "frame_id: map\nfixed_bucket_transit:\n  waypoints:\n"
      "    - {x: -1.55, y: 0.15, outer_y_passage: true}\n";
    rclcpp::NodeOptions options;
    options.enable_rosout(false).start_parameter_services(false);
    options.parameter_overrides({rclcpp::Parameter("fixed_bucket_routes_file", file_)});
    node_ = std::make_shared<rclcpp::Node>("bucket_passage_test", options);
    tf_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    blackboard_ = BT::Blackboard::create();
    blackboard_->set("node", node_);
    blackboard_->set("tf_buffer", tf_);
    factory_.registerFromPlugin(TEST_PLUGIN_PATH);
    tree_ = factory_.createTreeFromText(
      "<root BTCPP_format=\"4\"><BehaviorTree ID=\"Main\">"
      "<RemovePassedBucketGoals input_goals=\"{goals}\" output_goals=\"{goals}\"/>"
      "</BehaviorTree></root>", blackboard_);
  }
  void TearDown() override {std::filesystem::remove(file_);}
  void pose(double x, double y, double age = 0.)
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.header.stamp = node_->now() - rclcpp::Duration::from_seconds(age);
    transform.transform.translation.x = x;
    transform.transform.translation.y = y;
    transform.transform.rotation.z = 1.;  // yaw=pi, right field
    transform.transform.rotation.w = 0.;
    tf_->clear();
    ASSERT_TRUE(tf_->setTransform(transform, "test", false));
  }
  void goals(double x = 1.55, double y = .15, std::string frame = "map")
  {
    Goals values(3);
    for (auto & value : values) {value.header.frame_id = frame;}
    values[0].pose.position.x = x;
    values[0].pose.position.y = y;
    values[1].pose.position.x = .8;
    values[1].pose.position.y = .5;
    values[2].pose.position.x = .8;
    values[2].pose.position.y = 1.28;
    blackboard_->set("goals", values);
  }
  using Goals = std::vector<geometry_msgs::msg::PoseStamped>;
  Goals remaining() {return blackboard_->get<Goals>("goals");}
  std::string file_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  BT::Blackboard::Ptr blackboard_;
  BT::BehaviorTreeFactory factory_;
  BT::Tree tree_;
};

TEST_F(BucketTree, LoadPluginAndDropOnlyTaggedPassedTransit)
{
  goals(); pose(1.8, .3);
  EXPECT_EQ(tree_.tickOnce(), BT::NodeStatus::SUCCESS);
  ASSERT_EQ(remaining().size(), 2u);
  EXPECT_DOUBLE_EQ(remaining()[0].pose.position.x, .8);
  // Repeated BT ticks never put a passed gate back into the route.
  tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 2u);
}

TEST_F(BucketTree, RetainsNarrowGateFinalGoalAndNewRequests)
{
  goals(.8, .5); pose(1.02, .65);
  tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
  pose(.8, 1.28);
  Goals final(1); final[0].header.frame_id = "map";
  final[0].pose.position.x = .8; final[0].pose.position.y = 1.28;
  blackboard_->set("goals", final);
  tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 1u);
  goals(); pose(1.8, .05);  // replacement/retry is a new goal sequence
  tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}

TEST_F(BucketTree, MissingStaleTfAndFrameMismatchDoNotProvePassage)
{
  goals(); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
  pose(1.8, .3, .5); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
  pose(1.8, .3); goals(1.55, .15, "odom"); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}

TEST_F(BucketTree, SprintSamplesCrossNarrowDiskWithoutReturningToGate)
{
  goals(.8, .5);
  auto route = remaining();
  route[1].pose.position.y = .95;
  blackboard_->set("goals", route);
  pose(.8, .3, .10); tree_.tickOnce();
  ASSERT_EQ(remaining().size(), 3u);
  pose(.8, .7); tree_.tickOnce();
  ASSERT_EQ(remaining().size(), 2u);
  EXPECT_DOUBLE_EQ(remaining().front().pose.position.y, .95);
  // A duplicate transform cannot reuse the segment to drop further gates.
  tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 2u);
}

TEST_F(BucketTree, SweptPassageDoesNotEnlargeNarrowDisk)
{
  goals(.8, .5); pose(1., .3, .10); tree_.tickOnce();
  pose(1., .7); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}

TEST_F(BucketTree, SweptPassageRejectsLocalizationJumpAndLongGap)
{
  goals(.8, .5); pose(.8, .1, .10); tree_.tickOnce();
  pose(.8, .9); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
  pose(.8, .3, .25); tree_.tickOnce();
  pose(.8, .7); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}

TEST_F(BucketTree, MissingTransformBreaksSweepHistory)
{
  goals(.8, .5); pose(.8, .3, .10); tree_.tickOnce();
  tf_->clear(); tree_.tickOnce();
  pose(.8, .7); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}

TEST_F(BucketTree, ReplacementRequestCannotInheritSweptPassage)
{
  goals(.8, .5); pose(.8, .3, .10); tree_.tickOnce();
  auto replacement = remaining();
  replacement.front().header.stamp = node_->now();
  blackboard_->set("goals", replacement);
  pose(.8, .7); tree_.tickOnce();
  EXPECT_EQ(remaining().size(), 3u);
}
