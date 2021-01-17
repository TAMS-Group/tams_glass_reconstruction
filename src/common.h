#pragma once

#include <camera_calibration_parsers/parse_yml.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen_conversions/eigen_msg.h>
#include <fstream>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <opencv2/bgsegm.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>
#include <yaml-cpp/yaml.h>

struct Calibration {
  Eigen::Isometry3d tip_to_camera = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d base_to_object = Eigen::Isometry3d::Identity();
};

#define ASSERT(x)                                                              \
  if (!(x)) {                                                                  \
    throw std::runtime_error(#x);                                              \
  }

void loadCalibration(Calibration &calibration, const std::string &path) {

  YAML::Node data = YAML::LoadFile(path);

  auto readPose = [&](YAML::Node node) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    quat.x() = node["orientation"]["x"].as<double>();
    quat.y() = node["orientation"]["y"].as<double>();
    quat.z() = node["orientation"]["z"].as<double>();
    quat.w() = node["orientation"]["w"].as<double>();

    pose = Eigen::AngleAxisd(quat);

    pose.translation().x() = node["position"]["x"].as<double>();
    pose.translation().y() = node["position"]["y"].as<double>();
    pose.translation().z() = node["position"]["z"].as<double>();

    return pose;
  };

  calibration.tip_to_camera = readPose(data["tip_to_camera"]);
  calibration.base_to_object = readPose(data["base_to_object"]);

  ROS_INFO_STREAM("tip_to_camera\n" << calibration.tip_to_camera.matrix());
  ROS_INFO_STREAM("base_to_object\n" << calibration.base_to_object.matrix());
}
