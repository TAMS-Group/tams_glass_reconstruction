#include "common.h"

#include <pcl/point_types.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

struct VoxelGrid {
  Eigen::Vector3d mCenter;
  size_t mSize;
  double mResolution;
  std::vector<double> mData;
  double mDummy = 0.0;

public:
  VoxelGrid(const Eigen::Vector3d &center, size_t size, double resolution)
      : mCenter(center), mSize(size), mResolution(resolution),
        mData(size * size * size, 0.0) {}
  inline size_t dataIndex(size_t ix, size_t iy, size_t iz) const {
    size_t i = iz;
    i *= mSize;
    i += iy;
    i *= mSize;
    i += ix;
    return i;
  }
  const std::vector<double> &data() const { return mData; }
  inline size_t dataIndex(const Eigen::Vector3i &index) const {
    return dataIndex(index.x(), index.y(), index.z());
  }
  inline size_t size() const { return mSize; }
  inline double resolution() const { return mResolution; }
  inline double at(ssize_t ix, ssize_t iy, ssize_t iz) const {
    return mData[dataIndex(ix, iy, iz)];
  }
  inline double &at(ssize_t ix, ssize_t iy, ssize_t iz) {
    return mData[dataIndex(ix, iy, iz)];
  }
  inline double at(const Eigen::Vector3i &indices) const {
    return at(indices.x(), indices.y(), indices.z());
  }
  inline double &at(const Eigen::Vector3i &indices) {
    return at(indices.x(), indices.y(), indices.z());
  }
  inline Eigen::Vector3i index(const Eigen::Vector3d &position) const {
    Eigen::Vector3d pos = position - mCenter;
    pos *= 1.0 / mResolution;
    pos.array() += mSize * 0.5;
    pos.array().round();
    return pos.cast<int>();
  }
  inline Eigen::Vector3d position(const Eigen::Vector3i &indices) const {
    return ((indices.cast<double>().array() - mSize * 0.5) * mResolution)
               .matrix() +
           mCenter;
  }
  inline Eigen::Vector3d position(int ix, int iy, int iz) const {
    return position(Eigen::Vector3i(ix, iy, iz));
  }
  inline bool checkIndices(ssize_t ix, ssize_t iy, ssize_t iz) const {
    if (ix < 0 || iy < 0 || iz < 0)
      return false;
    if (ix >= mSize || iy >= mSize || iz >= mSize)
      return false;
    return true;
  }
  inline bool checkIndices(const Eigen::Vector3i &index) const {
    return checkIndices(index.x(), index.y(), index.z());
  }
  inline const Eigen::Vector3d &center() const { return mCenter; }
};

class Timer {
  const char *mName = nullptr;
  ros::WallTime mStartTime;

public:
  Timer(const char *name) : mName(name) { mStartTime = ros::WallTime::now(); }
  ~Timer() {
    auto stopTime = ros::WallTime::now();
    ROS_INFO_STREAM_THROTTLE(1.0, "time " << mName << " "
                                          << (stopTime - mStartTime));
  }
};

int main(int argc, char **argv) {

  ros::init(argc, argv, "tams_glass_reconstruct", 1);

  ros::NodeHandle node("~");

  ros::AsyncSpinner spinner(4);
  spinner.start();

  if (argc < 2) {
    ROS_ERROR_STREAM("usage: tams_glass reconstruct <data.bag>");
    return -1;
  }

  static const double gamma = 1.2;
  static const double huber_delta = 0.05;
  static const double smoothness = 0;
  static const double regularization = 10;
  static const double penalty = 10;
  static const int grid_size = 70;
  static const double grid_resolution = 0.005;
  static const double voxel_threshold = 0.015;
  static const double mesh_threshold = 0.015;
  static const double symmetry = 200;
  static const size_t ray_count = 1000;

  size_t iteration = 0;

  static const auto huberLoss = [](double x) {
    if (std::abs(x) <= huber_delta) {
      return 0.5 * x * x;
    } else {
      return huber_delta * (std::abs(x) - 0.5 * huber_delta);
    }
  };

  mkdir("./meshes", 0777);
  std::string output_prefix;
  if (auto *tok = strrchr(argv[1], '/')) {
    output_prefix = tok;
  } else {
    output_prefix = argv[1];
  }
  while (output_prefix.size() && output_prefix[0] == '/') {
    output_prefix = output_prefix.substr(1);
  }
  output_prefix = "./meshes/" + output_prefix;
  ROS_INFO_STREAM("output prefix " << output_prefix);

  static const auto computeWeight = [](double x) {
    if (x * x != 0.0) {
      return std::sqrt(huberLoss(x) / (x * x));
    } else {
      return 0.0;
    }
  };

  std::string tip_link = "ur5_tool0";
  std::string base_link = "table_top";

  moveit::planning_interface::MoveGroupInterface move_group("arm");

  moveit::core::RobotState robot_state = *move_group.getCurrentState();

  sensor_msgs::CameraInfo camera_info;
  std::string camera_name;
  if (!camera_calibration_parsers::readCalibrationYml(
          ros::package::getPath("tams_glass") + "/config/camera.yaml",
          camera_name, camera_info)) {
    ROS_ERROR_STREAM("failed to read camera");
    return -1;
  }

  image_geometry::PinholeCameraModel camera;
  if (!camera.fromCameraInfo(camera_info)) {
    ROS_ERROR_STREAM("failed to load camera info");
    return -1;
  }

  ros::Publisher vis_pub = node.advertise<visualization_msgs::MarkerArray>(
      "/tams_glass/visualization", 10);
  auto visualizeVoxelGrid = [&](const VoxelGrid &voxel_grid) {
    visualization_msgs::MarkerArray marker_array;

    std::string iteration_output_prefix = output_prefix + ".s" +
                                          std::to_string(symmetry) + ".i" +
                                          std::to_string(iteration);

    {
      visualization_msgs::Marker marker;
      marker.type = visualization_msgs::Marker::CUBE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.header.frame_id = "/world";
      marker.ns = "transparent";
      marker.pose.orientation.w = 1.0;
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 1.0;
      marker.color.a = 0.0;
      marker.scale.x = voxel_grid.resolution();
      marker.scale.y = voxel_grid.resolution();
      marker.scale.z = voxel_grid.resolution();

      double lo = std::numeric_limits<double>::max();
      double hi = 0.00000001;
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            hi = std::max(hi, voxel_grid.at(ix, iy, iz));
            lo = std::min(lo, voxel_grid.at(ix, iy, iz));
          }
        }
      }

      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            Eigen::Vector3d pos = voxel_grid.position(ix, iy, iz);

            double a = voxel_grid.at(ix, iy, iz);
            a = std::pow(a, gamma);

            if (!(a > 0.001)) {
              continue;
            }
            marker.points.emplace_back();
            marker.points.back().x = pos.x();
            marker.points.back().y = pos.y();
            marker.points.back().z = pos.z();
            marker.colors.emplace_back();
            marker.colors.back().r = 0;
            marker.colors.back().g = 0;
            marker.colors.back().b = 0;
            marker.colors.back().a = a * 3;
          }
        }
      }
      marker_array.markers.push_back(marker);
    }

    {
      std::ofstream stream(iteration_output_prefix + ".volume.bvox");
      auto writeInt = [&](uint32_t i) {
        stream.write((const char *)&i, sizeof(i));
      };
      auto writeFloat = [&](float i) {
        stream.write((const char *)&i, sizeof(i));
      };
      writeInt(grid_size);
      writeInt(grid_size);
      writeInt(grid_size);
      writeInt(1);
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            writeFloat(voxel_grid.at(ix, iy, iz));
          }
        }
      }
    }

    {
      std::ofstream stream(iteration_output_prefix + ".volume.raw");
      double hi = 0.0000001;
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            hi = std::max(hi, voxel_grid.at(ix, iy, iz));
          }
        }
      }
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            uint8_t v = std::max(
                0.0, std::min(255.0, voxel_grid.at(ix, iy, iz) * 255.0 / hi));
            stream.write((char *)&v, 1);
          }
        }
      }
    }

    {
      std::ofstream stream(iteration_output_prefix + ".points.xyz");

      visualization_msgs::Marker marker;
      marker.type = visualization_msgs::Marker::CUBE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.header.frame_id = "/world";
      marker.ns = "voxels";
      marker.pose.orientation.w = 1.0;
      marker.color.r = 0.9;
      marker.color.g = 0.9;
      marker.color.b = 0.9;
      marker.color.a = 1.0;
      marker.scale.x = voxel_grid.resolution();
      marker.scale.y = voxel_grid.resolution();
      marker.scale.z = voxel_grid.resolution();
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            Eigen::Vector3d pos = voxel_grid.position(ix, iy, iz);

            const int d = 1;
            std::array<double, (d + d + 1) * (d + d + 1) * (d + d + 1)>
                sample_buffer;
            size_t sample_count = 0;
            for (int dz = -d; dz <= d; dz++) {
              for (int dy = -d; dy <= d; dy++) {
                for (int dx = -d; dx <= d; dx++) {
                  Eigen::Vector3i p(ix + dx, iy + dy, iz + dz);
                  if (!voxel_grid.checkIndices(p)) {
                    continue;
                  }
                  sample_buffer[sample_count] = voxel_grid.at(p);
                  sample_count++;
                }
              }
            }
            std::sort(sample_buffer.begin(),
                      sample_buffer.begin() + sample_count);
            double a = sample_buffer[sample_count / 2 + 1];

            if (!(a > voxel_threshold)) {
              continue;
            }

            stream << pos.x() << " " << pos.y() << " " << pos.z() << "\n";

            marker.points.emplace_back();
            marker.points.back().x = pos.x();
            marker.points.back().y = pos.y();
            marker.points.back().z = pos.z();
          }
        }
      }
      marker_array.markers.push_back(marker);
    }

    {
      ROS_INFO_STREAM("meshing");

      pcl::PointCloud<pcl::PointNormal>::Ptr cloud(
          new pcl::PointCloud<pcl::PointNormal>);

      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            if (ix == 0 || iy == 0 || iz == 0 || ix + 1 == voxel_grid.size() ||
                iy + 1 == voxel_grid.size() || iz + 1 == voxel_grid.size()) {
              Eigen::Vector3d pos = voxel_grid.position(ix, iy, iz);
              pcl::PointNormal point;
              point.x = pos.x();
              point.y = pos.y();
              point.z = pos.z();
              point.normal_x = 0;
              point.normal_y = 0;
              point.normal_z = 1;
              cloud->push_back(point);
            }
          }
        }
      }

      ROS_INFO_STREAM(cloud->size() << " points");

      visualization_msgs::Marker marker;
      marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.header.frame_id = "/world";
      marker.ns = "mesh";
      marker.pose.orientation.w = 1.0;
      marker.color.r = 1;
      marker.color.g = 1;
      marker.color.b = 1;
      marker.color.a = 1.0;
      marker.scale.x = 1.0;
      marker.scale.y = 1.0;
      marker.scale.z = 1.0;

      if (cloud->size() >= 3) {

        struct MatchingCubesVoxelGrid
            : public pcl::MarchingCubes<pcl::PointNormal> {
          const VoxelGrid &voxel_grid;
          MatchingCubesVoxelGrid(const VoxelGrid &voxel_grid)
              : voxel_grid(voxel_grid) {}
          void voxelizeData() override {
#pragma omp parallel for
            for (int x = 0; x < res_x_; x++) {
              for (int y = 0; y < res_y_; y++) {
                for (int z = 0; z < res_z_; z++) {
                  Eigen::Vector3d point;
                  point[0] = min_p_[0] + (max_p_[0] - min_p_[0]) * x / res_x_;
                  point[1] = min_p_[1] + (max_p_[1] - min_p_[1]) * y / res_y_;
                  point[2] = min_p_[2] + (max_p_[2] - min_p_[2]) * z / res_z_;
                  Eigen::Vector3i index = voxel_grid.index(point);
                  if (voxel_grid.checkIndices(index)) {

                    int ix = index.x();
                    int iy = index.y();
                    int iz = index.z();

                    const int d = 1;
                    std::array<double, (d + d + 1) * (d + d + 1) * (d + d + 1)>
                        sample_buffer;
                    size_t sample_count = 0;
                    for (int dz = -d; dz <= d; dz++) {
                      for (int dy = -d; dy <= d; dy++) {
                        for (int dx = -d; dx <= d; dx++) {
                          Eigen::Vector3i p(ix + dx, iy + dy, iz + dz);
                          if (!voxel_grid.checkIndices(p)) {
                            continue;
                          }
                          sample_buffer[sample_count] = voxel_grid.at(p);
                          sample_count++;
                        }
                      }
                    }
                    std::sort(sample_buffer.begin(),
                              sample_buffer.begin() + sample_count);
                    double a = sample_buffer[sample_count / 2];

                    grid_[x * res_y_ * res_z_ + y * res_z_ + z] =
                        mesh_threshold - a;

                  } else {
                    grid_[x * res_y_ * res_z_ + y * res_z_ + z] = 1.0;
                  }
                }
              }
            }
          }
        };
        MatchingCubesVoxelGrid voxelizer(voxel_grid);
        voxelizer.setGridResolution(grid_size, grid_size, grid_size);
        voxelizer.setInputCloud(cloud);
        voxelizer.setPercentageExtendGrid(0.0);

        pcl::PointCloud<pcl::PointNormal> mesh_vertices;
        std::vector<pcl::Vertices> mesh_polygons;
        voxelizer.reconstruct(mesh_vertices, mesh_polygons);

        for (auto &polygon : mesh_polygons) {
          for (size_t index : polygon.vertices) {
            pcl::PointNormal vertex = mesh_vertices[index];
            marker.points.emplace_back();
            marker.points.back().x = vertex.x;
            marker.points.back().y = vertex.y;
            marker.points.back().z = vertex.z;
            if (!std::isfinite(marker.points.back().x))
              marker.points.back().x = 0.0;
            if (!std::isfinite(marker.points.back().y))
              marker.points.back().y = 0.0;
            if (!std::isfinite(marker.points.back().z))
              marker.points.back().z = 0.0;
          }
        }
        ROS_INFO_STREAM(mesh_polygons.size() << " polygons");

        {
          std::ofstream stream(iteration_output_prefix + ".mesh.obj");
          for (auto &p : mesh_vertices) {
            stream << "v " << p.x << " " << p.y << " " << p.z << "\n";
          }
          for (auto &polygon : mesh_polygons) {
            stream << "f " << (polygon.vertices[0] + 1) << " "
                   << (polygon.vertices[1] + 1) << " "
                   << (polygon.vertices[2] + 1) << "\n";
          }
        }
      }

      marker_array.markers.push_back(marker);
    }

    vis_pub.publish(marker_array);
  };

  Calibration calibration;
  loadCalibration(calibration, ros::package::getPath("tams_glass") +
                                   "/config/calibration.yaml");

  static moveit::core::RobotState static_robot_state = robot_state;
  struct Frame {
    sensor_msgs::JointState joint_state;
    cv::Mat raw_image, glass_image, mask_image;
    moveit::core::RobotState robot_state = static_robot_state;
  };
  std::deque<Frame> frames;

  {
    rosbag::Bag bag;
    bag.open(argv[1], rosbag::bagmode::Read);
    sensor_msgs::JointState last_joint_state;

    for (const rosbag::MessageInstance &bag_message : rosbag::View(bag)) {

      if (!ros::ok()) {
        ROS_ERROR_STREAM("canceled");
        return -1;
      }

      if (bag_message.getTopic() == "/joint_states") {
        if (auto joint_state_message =
                bag_message.instantiate<sensor_msgs::JointState>()) {
          last_joint_state = *joint_state_message;
        }
      }

      if (bag_message.getTopic() == "/camera_snapshot/ir/image") {
        if (auto image_message =
                bag_message.instantiate<sensor_msgs::Image>()) {

          auto &joint_state = last_joint_state;
          for (size_t i = 0; i < joint_state.name.size(); i++) {
            robot_state.setVariablePosition(joint_state.name[i],
                                            joint_state.position[i]);
          }
          robot_state.update(true);
          robot_state.updateLinkTransforms();

          Eigen::Affine3d camera_pose =
              robot_state.getFrameTransform(tip_link) *
              calibration.tip_to_camera;

          ROS_INFO_STREAM("camera position "
                          << camera_pose.translation().x() << " "
                          << camera_pose.translation().y() << " "
                          << camera_pose.translation().z());

          if (camera_pose.translation().z() < 0.9) {
            continue;
          }

          auto cv_image = cv_bridge::toCvCopy(image_message);

          frames.emplace_back();
          cv_image->image.convertTo(frames.back().raw_image, CV_32FC1);
          frames.back().joint_state = last_joint_state;
          frames.back().robot_state = robot_state;
        }
      }
    }
  }

  auto highpass = [&](cv::Mat src) {
    cv::Mat dst(src.rows, src.cols, src.type());

    for (int y = 0; y < src.rows; y++) {
      for (int x = 0; x < src.cols; x++) {

        int r = 3;

        double average = 0.0;
        double sample_count = 0.0;
        for (int y2 = std::max(0, y - r); y2 <= std::min(y + r, src.rows - 1);
             y2++) {
          for (int x2 = std::max(0, x - r); x2 <= std::min(x + r, src.cols - 1);
               x2++) {
            double v = src.at<float>(y2, x2);
            average += v;
            sample_count++;
          }
        }
        average /= sample_count;

        double variance = 0.0;
        for (int y2 = std::max(0, y - r); y2 <= std::min(y + r, src.rows - 1);
             y2++) {
          for (int x2 = std::max(0, x - r); x2 <= std::min(x + r, src.cols - 1);
               x2++) {
            double v = src.at<float>(y2, x2);
            double d = v - average;
            variance += d * d;
          }
        }
        variance /= sample_count;

        dst.at<float>(y, x) = variance;
      }
    }

    return dst;
  };

  auto peaks = [&](cv::Mat src) {
    cv::Mat dst(src.rows, src.cols, src.type());
    for (int y = 0; y < src.rows; y++) {
      for (int x = 0; x < src.cols; x++) {
        dst.at<float>(y, x) = 1.0;
        int r = 1;
        for (int y2 = std::max(0, y - r); y2 <= std::min(y + r, src.rows - 1);
             y2++) {
          for (int x2 = std::max(0, x - r); x2 <= std::min(x + r, src.cols - 1);
               x2++) {
            if (src.at<float>(y2, x2) > src.at<float>(y, x)) {
              dst.at<float>(y, x) = 0.0;
            }
          }
        }
        if (dst.at<float>(y, x) == 1.0) {
          dst.at<float>(y, x) = src.at<float>(y, x);
        }
      }
    }
    return dst;
  };

  auto highpass2 = [&](cv::Mat src) {
    cv::Mat blurred;
    // cv::medianBlur(src, blurred, 7);
    cv::GaussianBlur(src, blurred, cv::Size(0, 0), 0.5, 0.5);
    return cv::abs(src - blurred);
  };

  ROS_INFO_STREAM(__LINE__);

  auto makeMask = [&](cv::Mat image) {
    Timer t("make mask");

    {
      double lo, hi;
      cv::minMaxIdx(image, &lo, &hi);
      image = (image - lo) / (hi - lo);
    }

    cv::medianBlur(image, image, 3);

    for (int y = 0; y < image.rows; y++) {
      for (int x = 0; x < image.cols; x++) {
        image.at<float>(y, x) = (image.at<float>(y, x) > 0.1 ? 1.0 : 0.0);
      }
    }

    cv::erode(image, image, cv::Mat());

    return image;
  };

  auto detectGlass = [&](cv::Mat image) {

    cv::Mat image0 = image * 1.0;

    Timer t("detect glass");

    {
      double lo, hi;
      cv::minMaxIdx(image, &lo, &hi);
      image = (image - lo) / (hi - lo);
    }

    for (int i = 0; i < 3; i++) {
      image = highpass2(image);
    }

    {
      cv::Mat blurred;

      cv::resize(image, blurred, cv::Size(0, 0), 0.25, 0.25);

      cv::blur(blurred, blurred, cv::Size(10, 10));

      int dilation_size = 2;
      cv::Mat element = cv::getStructuringElement(
          cv::MORPH_ELLIPSE,
          cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
          cv::Point(dilation_size, dilation_size));
      for (int i = 0; i < 5; i++) {
        cv::dilate(blurred, blurred, element);
      }

      cv::blur(blurred, blurred, cv::Size(10, 10));

      cv::resize(blurred, blurred, cv::Size(0, 0), 4, 4);

      image /= blurred;
    }

    image *= 127;
    image.convertTo(image, CV_8U);

    cv::GaussianBlur(image, image, cv::Size(0, 0), 2.0, 2.0);

    cv::medianBlur(image, image, 21);

    image.convertTo(image, CV_32FC1);

    image *= 1.0 / 255;

    image *= 2.0;

    image *= 2.0;

    image -= 0.2;

    image = cv::max(image, 0.0);
    image = cv::min(image, 1.0);

    image = 1.0 - image;

    if (1) {
      cv::Mat mask;
      cv::threshold(image, mask, 0.3, 255, cv::THRESH_BINARY);
      mask.convertTo(mask, CV_8U);
      // cv::imshow("mask", mask);
      cv::Mat labels, stats, centroids;
      cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8,
                                       CV_32S);
      for (int component = 1; component < stats.rows; component++) {
        int border = 3;
        bool removeThisComponent = false;
        if (stats.at<int>(component, cv::CC_STAT_LEFT) <= border) {
          removeThisComponent = true;
        }
        if (stats.at<int>(component, cv::CC_STAT_TOP) <= border) {
          removeThisComponent = true;
        }
        if ((stats.at<int>(component, cv::CC_STAT_WIDTH) +
             stats.at<int>(component, cv::CC_STAT_LEFT)) >=
            image.cols - border) {
          removeThisComponent = true;
        }
        if ((stats.at<int>(component, cv::CC_STAT_HEIGHT) +
             stats.at<int>(component, cv::CC_STAT_TOP)) >=
            image.rows - border) {
          removeThisComponent = true;
        }
        if (removeThisComponent) {
          cv::Mat mask(image.rows, image.cols, CV_32F, cv::Scalar(1, 1, 1, 1));
          for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
              if (labels.at<int>(y, x) == component) {
                mask.at<float>(y, x) = 0.0;
              }
            }
          }
          {
            int dilation_size = 2;
            cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                cv::Point(dilation_size, dilation_size));
            for (int i = 0; i < 5; i++) {
              cv::erode(mask, mask, element);
            }
          }
          image = image.mul(mask);
        }
      }
    }

    image = cv::max(image, 0.0);
    image = cv::min(image, 1.0);

    return image;
  };

  ROS_INFO_STREAM(__LINE__);

  if (0) {
    // preview
    for (auto &frame : frames) {

      {
        cv::Mat raw;
        frame.raw_image.copyTo(raw);
        double lo, hi;
        cv::minMaxIdx(raw, &lo, &hi);
        raw = (raw - lo) / (hi - lo);
        cv::imshow("raw", raw);
      }

      frame.raw_image.copyTo(frame.glass_image);
      cv::imshow("image", detectGlass(frame.glass_image));

      frame.raw_image.copyTo(frame.mask_image);
      cv::imshow("mask", makeMask(frame.mask_image));

      cv::waitKey(0);

      //  break;
    }
  }

  {
    size_t n = frames.size();
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      frames[i].raw_image.copyTo(frames[i].glass_image);
      frames[i].glass_image = detectGlass(frames[i].glass_image);

      frames[i].raw_image.copyTo(frames[i].mask_image);
      frames[i].mask_image = makeMask(frames[i].mask_image);
    }
  }

  ROS_INFO_STREAM(__LINE__);

  VoxelGrid voxel_grid(
      frames.back().robot_state.getFrameTransform(base_link) *
              calibration.base_to_object * Eigen::Vector3d::Zero() +
          Eigen::Vector3d(0, 0, grid_size * grid_resolution * 0.5) +
          Eigen::Vector3d(0, 0, 0.005),
      grid_size, grid_resolution);

  ROS_INFO_STREAM(__LINE__);

  if (0) {

    // simple

    ROS_INFO_STREAM("simple reconstruction");

    for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
      for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
        for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
          voxel_grid.at(ix, iy, iz) = 0;
        }
      }
    }

    int used_frames = 0;

    for (size_t frame_index = 0; frame_index < frames.size();
         frame_index++ /*frame_index =
             std::min(frame_index + 1, frames.size() - 1)*/) {
      auto &frame = frames[frame_index];

      ROS_INFO_STREAM("frame " << frame_index << " / " << frames.size());

      cv::Mat glass_image = frame.glass_image;
      cv::Mat mask_image = frame.mask_image;

      glass_image = glass_image.mul(mask_image);

      cv::imshow("original", glass_image);
      cv::imshow("mask", mask_image);

      glass_image.convertTo(glass_image, CV_8UC1, 255.0);

      // Threshold
      cv::threshold(glass_image, glass_image, 10, 255, cv::THRESH_BINARY);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(glass_image, contours, CV_RETR_LIST,
                       CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
      std::vector<std::pair<double, size_t>> contour_areas;
      for (size_t contourIndex = 0; contourIndex < contours.size();
           contourIndex++) {
        contour_areas.emplace_back(cv::contourArea(contours[contourIndex]),
                                   contourIndex);
      }
      std::sort(contour_areas.begin(), contour_areas.end());
      std::reverse(contour_areas.begin(), contour_areas.end());
      if (contour_areas.empty()) {
        ROS_INFO_STREAM("no transparent areas found in this image");
        continue;
      }
      if (contour_areas.size() >= 2) {
        if (contour_areas[0].first < contour_areas[1].first * 4) {
          ROS_INFO_STREAM("skipped");
          continue;
        }
      }
      ROS_INFO_STREAM("contour area " << contour_areas[0].first);
      if (contour_areas[0].first < 5000) {
        ROS_INFO_STREAM("too small");
        continue;
      }
      size_t selected_contour = contour_areas[0].second;
      glass_image *= 0;
      std::vector<cv::Point> hull;
      cv::convexHull(contours[selected_contour], hull);

      cv::fillPoly(glass_image, std::vector<std::vector<cv::Point>>({hull}),
                   cv::Scalar(255, 255, 255, 255));

      // Visualize
      cv::imshow("image", glass_image);

      glass_image.convertTo(glass_image, CV_32FC1, 1.0 / 255.0);

      // Get camera info for projection
      Eigen::Affine3d camera_pose =
          frame.robot_state.getFrameTransform(tip_link) *
          calibration.tip_to_camera;
      Eigen::Affine3d camera_pose_inverse = camera_pose.inverse();

      // Project images into voxel field
      ROS_INFO("image");
      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            Eigen::Vector3d position = voxel_grid.position(ix, iy, iz);
            Eigen::Vector3d pos = camera_pose_inverse * position;
            cv::Point2d p =
                camera.project3dToPixel(cv::Point3d(pos.x(), pos.y(), pos.z()));
            int px = std::round(p.x);
            int py = std::round(p.y);
            if (px < 0 || py < 0 || px >= glass_image.cols ||
                py >= glass_image.rows) {
              continue;
            }
            float v = glass_image.at<float>(py, px);
            if (v > 0.5) {
              voxel_grid.at(ix, iy, iz) += 1;
            }
          }
        }
      }

      used_frames++;
    }

    double vmax = 0;
    for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
      for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
        for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
          vmax = std::max(vmax, voxel_grid.at(ix, iy, iz));
        }
      }
    }

    // Threshold ?
    for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
      for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
        for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
          voxel_grid.at(ix, iy, iz) =
              ((voxel_grid.at(ix, iy, iz) >= vmax * 0.9) ? 1.0 : 0.0);
        }
      }
    }

    // Visualize
    visualizeVoxelGrid(voxel_grid);

  } else {

    // optimization

    size_t variable_count =
        voxel_grid.size() * voxel_grid.size() * voxel_grid.size();

    for (;; iteration++) {

      ROS_INFO_STREAM("iteration " << iteration);

      auto iteration_begin_time = ros::Time::now();

      typedef float Real;

      std::vector<Eigen::Triplet<Real>> gradients;
      std::vector<Real> residuals;

      for (auto &frame : frames) {

        cv::Mat glass_image = frame.glass_image;
        cv::Mat mask_image = frame.mask_image;

        Eigen::Affine3d camera_pose =
            frame.robot_state.getFrameTransform(tip_link) *
            calibration.tip_to_camera;
        Eigen::Affine3d camera_pose_inverse = camera_pose.inverse();

        for (size_t ray_index = 0; ray_index < ray_count; ray_index++) {

          Eigen::Vector3d ray_point = voxel_grid.position(
              rand() % voxel_grid.size(), rand() % voxel_grid.size(),
              rand() % voxel_grid.size());

          double residual = 0.0;
          {
            auto pos = camera_pose_inverse * ray_point;
            cv::Point2d p =
                camera.project3dToPixel(cv::Point3d(pos.x(), pos.y(), pos.z()));
            int px = std::round(p.x);
            int py = std::round(p.y);
            if (px < 0 || py < 0 || px >= glass_image.cols ||
                py >= glass_image.rows) {
              continue;
            }
            residual = glass_image.at<float>(py, px);
            if (!(mask_image.at<float>(py, px) > 0.1)) {
              residual = 0.0;
            }
          }

          residual = std::pow(residual, 1.0 / gamma);

          Eigen::Vector3d ray_direction =
              (ray_point - camera_pose.translation()).normalized();

#if 1
          double weight = 0.0;
          {
            double current_sum = 0.0;
            int last_data_index = -1;
            for (int step_index = -(int)voxel_grid.size();
                 step_index <= (int)voxel_grid.size(); step_index++) {
              Eigen::Vector3d p =
                  ray_point + ray_direction * (voxel_grid.resolution() *
                                               (double)step_index);
              Eigen::Vector3i index = voxel_grid.index(p);
              if (!voxel_grid.checkIndices(index)) {
                continue;
              }
              int data_index = voxel_grid.dataIndex(index);
              if (data_index == last_data_index) {
                continue;
              }
              current_sum += voxel_grid.data()[data_index];
            }
            weight = computeWeight(current_sum - residual);
          }

          {
            int last_data_index = -1;
            for (int step_index = -(int)voxel_grid.size();
                 step_index <= (int)voxel_grid.size(); step_index++) {
              Eigen::Vector3d p =
                  ray_point + ray_direction * (voxel_grid.resolution() *
                                               (double)step_index);
              Eigen::Vector3i index = voxel_grid.index(p);
              if (!voxel_grid.checkIndices(index)) {
                continue;
              }
              int data_index = voxel_grid.dataIndex(index);
              if (data_index == last_data_index) {
                continue;
              }
              last_data_index = data_index;
              gradients.emplace_back(residuals.size(), data_index, weight);
              residual -= voxel_grid.at(index);
            }
          }
#endif

          residuals.emplace_back(residual * weight);
        }
      }

      for (size_t i = 0; i < variable_count; i++) {
        gradients.emplace_back(residuals.size(), i, 0.001);
        residuals.emplace_back(0.0);
      }

      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            size_t i = voxel_grid.dataIndex(Eigen::Vector3i(ix, iy, iz));
            double v = voxel_grid.at(ix, iy, iz);
            // double goal = ((v < 0.5) ? (-v) : (1.0 - v));
            double goal = -v;
            double weight = regularization * computeWeight(goal);
            // weight *= weight;
            gradients.emplace_back(residuals.size(), i, weight);
            residuals.emplace_back(goal * weight);
          }
        }
      }

      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            size_t i = voxel_grid.dataIndex(Eigen::Vector3i(ix, iy, iz));
            double v = voxel_grid.at(ix, iy, iz);
            double weight = penalty;
            double maximum = 0.9;
            if (v < 0.0) {
              gradients.emplace_back(residuals.size(), i, weight);
              residuals.emplace_back(-v * weight);
            }
            if (v >= maximum) {
              gradients.emplace_back(residuals.size(), i, weight);
              residuals.emplace_back((maximum - v) * weight);
            }
          }
        }
      }

      for (int iz = 0; iz < voxel_grid.size(); iz++) {
        for (int iy = 0; iy < voxel_grid.size(); iy++) {
          for (int ix = 0; ix < voxel_grid.size(); ix++) {
            size_t i = voxel_grid.dataIndex(Eigen::Vector3i(ix, iy, iz));
            const int d = 1;
            std::array<double, (d + d + 1) * (d + d + 1) * (d + d + 1)>
                sample_buffer;
            size_t sample_count = 0;
            for (int dz = -d; dz <= d; dz++) {
              for (int dy = -d; dy <= d; dy++) {
                for (int dx = -d; dx <= d; dx++) {
                  Eigen::Vector3i p(ix + dx, iy + dy, iz + dz);
                  if (!voxel_grid.checkIndices(p)) {
                    continue;
                  }
                  sample_buffer[sample_count] = voxel_grid.at(p);
                  sample_count++;
                }
              }
            }
            std::sort(sample_buffer.begin(),
                      sample_buffer.begin() + sample_count);
            double goal =
                sample_buffer[sample_count / 2] - voxel_grid.at(ix, iy, iz);
            double weight = smoothness * computeWeight(goal);
            gradients.emplace_back(residuals.size(), i, weight);
            residuals.emplace_back(goal * weight);
          }
        }
      }

      if (iteration >= 4 && symmetry > 0) {
        cv::Mat image(grid_size, grid_size, CV_32F, cv::Scalar(0.0));
        for (int iz = 0; iz < voxel_grid.size(); iz++) {
          for (int iy = 0; iy < voxel_grid.size(); iy++) {
            for (int ix = 0; ix < voxel_grid.size(); ix++) {
              image.at<float>(iy, ix) += voxel_grid.at(ix, iy, iz);
            }
          }
        }
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, grid_size, grid_size, grid_size,
                           grid_size, cv::BORDER_CONSTANT, cv::Scalar(0.0));
        {
          double lo, hi;
          cv::minMaxIdx(padded, &lo, &hi);
          padded *= 1.0 / hi;
        }
        cv::Mat flipped;
        cv::flip(image, flipped, -1);
        cv::Mat result;
        cv::matchTemplate(padded, flipped, result, cv::TM_CCORR);
        {
          double lo, hi;
          cv::minMaxIdx(result, &lo, &hi);
          result *= 1.0 / hi;
        }
        cv::Point maxLoc;
        cv::minMaxLoc(result, nullptr, nullptr, nullptr, &maxLoc);
        Eigen::Vector2d center = Eigen::Vector2d(maxLoc.x, maxLoc.y) * 0.5;
        {
          cv::Mat alignment(grid_size, grid_size, CV_32FC3,
                            cv::Scalar(0.0, 0.0, 0.0));
          for (int iy = 0; iy < grid_size; iy++) {
            for (int ix = 0; ix < grid_size; ix++) {
              float a = image.at<float>(iy, ix);
              int jx = std::max(
                  0, std::min((int)grid_size - 1,
                              (int)std::round(center.x() * 2 - ix - 1)));
              int jy = std::max(
                  0, std::min((int)grid_size - 1,
                              (int)std::round(center.y() * 2 - iy - 1)));
              float b = image.at<float>(jy, jx);
              alignment.at<cv::Vec3f>(iy, ix)[0] = a;
              alignment.at<cv::Vec3f>(iy, ix)[1] = b;
              alignment.at<cv::Vec3f>(iy, ix)[2] = a;
            }
          }
          cv::circle(result, maxLoc, 2.0, cv::Scalar(0.0));
          cv::circle(image,
                     cv::Point(std::round(center.x()), std::round(center.y())),
                     2.0, cv::Scalar(1.0));
          cv::circle(image,
                     cv::Point(std::round(center.x()), std::round(center.y())),
                     16.0, cv::Scalar(1.0));
          cv::resize(image, image, cv::Size(0, 0), 4.0, 4.0);
          cv::resize(padded, padded, cv::Size(0, 0), 4.0, 4.0);
          cv::resize(result, result, cv::Size(0, 0), 4.0, 4.0);
          cv::resize(alignment, alignment, cv::Size(0, 0), 4.0, 4.0);
          cv::imshow("image", image);
          cv::imshow("padded", padded);
          cv::imshow("result", result);
          cv::imshow("alignment", alignment);
          cv::waitKey(10);
        }
        {
          for (int iy = 0; iy < voxel_grid.size(); iy++) {
            for (int ix = 0; ix < voxel_grid.size(); ix++) {
              for (int iz = 0; iz < voxel_grid.size(); iz++) {
                double ref = 0.0;
                int count = 0;
                for (int angle = 30; angle < 360; angle += 30) {
                  Eigen::Vector2d v =
                      Eigen::Vector2d(ix - center.x(), iy - center.y());
                  v = Eigen::Rotation2D<double>(angle * M_PI / 180) * v;
                  int jx = (int)std::round(center.x() + v.x() - 1);
                  int jy = (int)std::round(center.y() + v.y() - 1);
                  if (jx < 0 || jy < 0 || jx >= grid_size || jy >= grid_size) {
                    continue;
                  }
                  count++;
                  ref += voxel_grid.at(jx, jy, iz);
                }
                if (count == 0) {
                  continue;
                }
                ref /= count;
                double goal = ref - voxel_grid.at(ix, iy, iz);
                double weight = computeWeight(goal) * symmetry;
                gradients.emplace_back(
                    residuals.size(), voxel_grid.dataIndex(ix, iy, iz), weight);
                residuals.emplace_back(goal * weight);
              }
            }
          }
        }
      }

      Eigen::SparseMatrix<Real> gradient_matrix(residuals.size(),
                                                variable_count);
      gradient_matrix.setFromTriplets(gradients.begin(), gradients.end());

      Eigen::Matrix<Real, Eigen::Dynamic, 1> residual_vector(residuals.size());
      for (size_t i = 0; i < residuals.size(); i++) {
        residual_vector[i] = residuals[i];
      }

      ROS_INFO_STREAM(residual_vector.size() << " objectives");
      ROS_INFO_STREAM(variable_count << " variables");

      ROS_INFO_STREAM("solving");
      Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<Real>> solver;
      solver.setMaxIterations(20);
      solver.compute(gradient_matrix);
      Eigen::Matrix<Real, Eigen::Dynamic, 1> solution_vector(variable_count);

      auto solve_begin_time = ros::Time::now();
      solution_vector = solver.solve(residual_vector);
      ROS_INFO_STREAM("solve time " << ros::Time::now() - solve_begin_time);

      for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
        for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
          for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
            voxel_grid.at(ix, iy, iz) +=
                solution_vector[voxel_grid.dataIndex(ix, iy, iz)];
          }
        }
      }

      ROS_INFO_STREAM("ready");

      ROS_INFO_STREAM("outer iteration time "
                      << ros::Time::now() - iteration_begin_time);

      visualizeVoxelGrid(voxel_grid);
    }
  }
}
