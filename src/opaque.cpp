#include "common.h"

int main(int argc, char **argv) {

  ros::init(argc, argv, "tams_glass_opaque", 1);

  ros::NodeHandle node("~");

  ros::AsyncSpinner spinner(4);
  spinner.start();

  if (argc < 2) {
    ROS_ERROR_STREAM("usage: tams_glass opaque <data.bag> [<outputfile>]");
    return -1;
  }

  ros::Publisher point_pub =
      node.advertise<sensor_msgs::PointCloud2>("/tams_glass/pointcloud", 10);

  float clip_radius = 0.3;
  float min_height = 0.54;

  std::string tip_link = "ur5_tool0";
  std::string base_link = "table_top";
  moveit::planning_interface::MoveGroupInterface move_group("arm");
  moveit::core::RobotState robot_state = *move_group.getCurrentState();

  Calibration calibration;
  loadCalibration(calibration, ros::package::getPath("tams_glass") +
                                   "/config/calibration.yaml");

  sensor_msgs::JointState last_joint_state;
  bool has_received_joint_state = false;

  sensor_msgs::PointCloud point_cloud;
  point_cloud.header.frame_id = "world";

  ROS_INFO_STREAM("processing");

  size_t max_point_count = 10000;

  uint64_t total_point_count = 0;

  std::mt19937 rng(0);

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
        last_joint_state = *joint_state_message;
        has_received_joint_state = true;
      }
    }

    if (bag_message.getTopic() == "/camera_snapshot/depth/points") {

      if (auto point_cloud_message =
              bag_message.instantiate<sensor_msgs::PointCloud2>()) {

        if (!has_received_joint_state) {
          continue;
        }

        auto &joint_state = last_joint_state;
        for (size_t i = 0; i < joint_state.name.size(); i++) {
          robot_state.setVariablePosition(joint_state.name[i],
                                          joint_state.position[i]);
        }
        robot_state.update(true);
        robot_state.updateLinkTransforms();

        Eigen::Affine3f camera_pose = (robot_state.getFrameTransform(tip_link) *
                                       calibration.tip_to_camera)
                                          .cast<float>();

        Eigen::Vector3f center =
            (robot_state.getFrameTransform(base_link) * Eigen::Vector3d::Zero())
                .cast<float>();

        for (sensor_msgs::PointCloud2ConstIterator<float> it(
                 *point_cloud_message, "x");
             it != it.end(); ++it) {
          Eigen::Vector3f point(it[0], it[1], it[2]);

          // point.z() += 0.015;
          point.z() += 0.017;

          point = camera_pose * point;

          if (!std::isfinite(point.x()) || !std::isfinite(point.y()) ||
              !std::isfinite(point.z())) {
            continue;
          }

          if (point.z() < min_height) {
            continue;
          }

          if ((point - center).squaredNorm() > clip_radius * clip_radius) {
            continue;
          }

          geometry_msgs::Point32 *msg_point = nullptr;
          if (point_cloud.points.size() >= max_point_count) {
            if (std::uniform_real_distribution<double>()(rng) <
                max_point_count * 1.0 / total_point_count) {
              msg_point = &point_cloud.points[rand() % max_point_count];
            }
          } else {
            point_cloud.points.emplace_back();
            msg_point = &point_cloud.points.back();
          }
          if (msg_point) {
            msg_point->x = point.x();
            msg_point->y = point.y();
            msg_point->z = point.z();
          }

          total_point_count++;
        }
      }
    }
  }

  {
    ROS_INFO_STREAM("writing file");
    {
      std::ofstream f(std::string(argv[1]) + ".pointcloud.xyz");
      for (auto &point : point_cloud.points) {
        f << point.x << " " << point.y << " " << point.z << "\n";
      }
    }
    ROS_INFO_STREAM("file written");
  }

  sensor_msgs::PointCloud2 point_cloud_2;
  sensor_msgs::convertPointCloudToPointCloud2(point_cloud, point_cloud_2);

  ROS_INFO_STREAM("ready");

  while (true) {
    ROS_INFO_STREAM("publishing");
    point_pub.publish(point_cloud_2);
    ros::Duration(2.0).sleep();
  }
}
