#include "common.h"

int main(int argc, char **argv) {

  ros::init(argc, argv, "tams_glass_poses", 0);

  ros::NodeHandle node("~");

  ros::AsyncSpinner spinner(4);
  spinner.start();

  if (argc < 2) {
    ROS_ERROR_STREAM("usage: tams_glass poses <data.bag>");
    return -1;
  }

  moveit::planning_interface::MoveGroupInterface move_group("arm");

  moveit::core::RobotState robot_state = *move_group.getCurrentState();

  ros::Publisher display_planned_path =
      node.advertise<moveit_msgs::DisplayTrajectory>(
          "/move_group/display_planned_path", 1, true);

  cv::Mat mat(256, 256, CV_32F, cv::Scalar(0.0));
  cv::imshow("Press key to continue", mat);

  ros::Time last_time;

  {
    rosbag::Bag bag;
    bag.open(argv[1], rosbag::bagmode::Read);

    for (const rosbag::MessageInstance &bag_message : rosbag::View(bag)) {

      if (!ros::ok()) {
        ROS_ERROR_STREAM("canceled");
        return -1;
      }

      if (bag_message.getTopic() == "/joint_states") {

        if (auto joint_state_message =
                bag_message.instantiate<sensor_msgs::JointState>()) {
          auto joint_state = *joint_state_message;

          if ((joint_state_message->header.stamp - last_time).toSec() < 3.0) {
            continue;
          }
          last_time = joint_state_message->header.stamp;

          for (size_t i = 0; i < joint_state.name.size(); i++) {
            robot_state.setVariablePosition(joint_state.name[i],
                                            joint_state.position[i]);
          }
          robot_state.update(true);
          robot_state.updateLinkTransforms();

          moveit_msgs::DisplayTrajectory msg;

          for (size_t i = 0; i < 2; i++) {
            msg.trajectory.emplace_back();
            msg.trajectory.back().joint_trajectory.joint_names =
                robot_state.getVariableNames();
            msg.trajectory.back().joint_trajectory.points.emplace_back();
            for (auto &n : msg.trajectory.back().joint_trajectory.joint_names) {
              msg.trajectory.back()
                  .joint_trajectory.points.back()
                  .positions.emplace_back(robot_state.getVariablePosition(n));
            }
          }

          msg.trajectory_start.joint_state = joint_state;

          display_planned_path.publish(msg);

          cv::waitKey(0);
        }
      }
    }
  }
}
