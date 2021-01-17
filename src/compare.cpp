#include "common.h"

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

static const double grid_resolution = 0.005;
static const double mesh_threshold = 0.002;
static const int grid_size = 70;
static const Eigen::Vector3d grid_center(0.918642, 0.704307, 0.708361);

struct MatchingCubesVoxelGrid : public pcl::MarchingCubes<pcl::PointNormal> {
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

            grid_[x * res_y_ * res_z_ + y * res_z_ + z] =
                (0.5 - voxel_grid.at(index));

          } else {
            grid_[x * res_y_ * res_z_ + y * res_z_ + z] = 1.0;
          }
        }
      }
    }
  }
};

void medianFilter(VoxelGrid &out) {
  VoxelGrid in = out;

  for (int ix = 0; ix < in.size(); ix++) {
    for (int iy = 0; iy < in.size(); iy++) {
      for (int iz = 0; iz < in.size(); iz++) {

        Eigen::Vector3i index(ix, iy, iz);

        const int d = 2;
        std::array<double, (d + d + 1) * (d + d + 1) * (d + d + 1)>
            sample_buffer;
        for (auto &v : sample_buffer) {
          v = 0.0;
        }
        size_t sample_count = 0;
        for (int dz = -d; dz <= d; dz++) {
          for (int dy = -d; dy <= d; dy++) {
            for (int dx = -d; dx <= d; dx++) {
              Eigen::Vector3i p(ix + dx, iy + dy, iz + dz);
              if (!in.checkIndices(p)) {
                continue;
              }
              sample_buffer[sample_count] = in.at(p);
              sample_count++;
            }
          }
        }
        std::sort(sample_buffer.begin(), sample_buffer.begin() + sample_count);
        double a = sample_buffer[sample_count * 3 / 4];

        out.at(ix, iy, iz) = a;
      }
    }
  }
}

void saveMesh(const VoxelGrid &voxel_grid, const std::string &path) {

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

  MatchingCubesVoxelGrid voxelizer(voxel_grid);
  voxelizer.setGridResolution(voxel_grid.size(), voxel_grid.size(),
                              voxel_grid.size());
  voxelizer.setInputCloud(cloud);
  voxelizer.setPercentageExtendGrid(0.0);
  pcl::PointCloud<pcl::PointNormal> mesh_vertices;
  std::vector<pcl::Vertices> mesh_polygons;
  voxelizer.reconstruct(mesh_vertices, mesh_polygons);
  {
    std::ofstream stream(path);
    for (auto &p : mesh_vertices) {
      stream << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }
    for (auto &polygon : mesh_polygons) {
      stream << "f " << (polygon.vertices[0] + 1) << " "
             << (polygon.vertices[1] + 1) << " " << (polygon.vertices[2] + 1)
             << "\n";
    }
  }
}

VoxelGrid loadDepthScan(const std::string &path) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
      std::stringstream stream(line);
      double x, y, z;
      if (stream >> x >> y >> z) {
        pcl::PointXYZ point;
        point.x = x;
        point.y = y;
        point.z = z;
        cloud->push_back(point);
      }
    }
  }

  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(0.01);
    outrem.setMinNeighborsInRadius(5);
    outrem.filter(*cloud);
  }

  {
    std::ofstream stream(std::string() + path + ".filtered.xyz");
    for (auto &point : cloud->points) {
      stream << point.x << " " << point.y << " " << point.z << "\n";
    }
  }

  VoxelGrid voxel_grid(grid_center, grid_size, grid_resolution);
  for (auto &point : cloud->points) {
    for (size_t i = 0; i < 10; i++) {
      auto index = voxel_grid.index(Eigen::Vector3d::Random() * 0.0 +
                                    Eigen::Vector3d(point.x, point.y, point.z));
      if (voxel_grid.checkIndices(index)) {
        voxel_grid.at(index) = 1.0;
      }
    }
  }

  // medianFilter(voxel_grid);

  saveMesh(voxel_grid, std::string() + path + ".filtered.obj");

  return voxel_grid;
}

VoxelGrid loadGlassReconstruction(const std::string &path) {

  std::ifstream voxel_file(path);

  if (!voxel_file) {
    throw std::runtime_error("failed to open voxel file");
  }

  struct {
    uint32_t size_x = 0, size_y = 1, size_z = 2, reserved = 0;
  } voxel_header;

  voxel_file.read((char *)&voxel_header, sizeof(voxel_header));

  if (voxel_header.size_x != voxel_header.size_y ||
      voxel_header.size_y != voxel_header.size_z) {
    throw std::runtime_error("currently only voxel cubes are supported");
  }

  VoxelGrid voxel_grid(grid_center, voxel_header.size_x, grid_resolution);

  double range = 0.0;

  for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
    for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
      for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
        float v = 0.0f;
        voxel_file.read((char *)&v, sizeof(v));
        voxel_grid.at(ix, iy, iz) = v;
        range = std::max(range, (double)v);
      }
    }
  }

  for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
    for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
      for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
        voxel_grid.at(ix, iy, iz) *= 1.0 / range;
      }
    }
  }

  // medianFilter(voxel_grid);

  saveMesh(voxel_grid, std::string() + path + ".filtered.obj");

  return voxel_grid;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxelPoints(const VoxelGrid &grid) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr ret(new pcl::PointCloud<pcl::PointXYZ>());
  for (size_t iz = 0; iz < grid.size(); iz++) {
    for (size_t iy = 0; iy < grid.size(); iy++) {
      for (size_t ix = 0; ix < grid.size(); ix++) {
        if (grid.at(ix, iy, iz) > 0.5) {
          auto pos = grid.position(ix, iy, iz);
          pcl::PointXYZ point;
          point.x = pos.x();
          point.y = pos.y();
          point.z = pos.z();
          ret->points.push_back(point);
        }
      }
    }
  }
  return ret;
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "tams_glass_compare", 0);

  VoxelGrid a = loadDepthScan(argv[1]);

  VoxelGrid b = loadGlassReconstruction(argv[2]);

  auto ca = voxelPoints(a);
  auto cb = voxelPoints(b);

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cb);
  icp.setInputTarget(ca);
  icp.setRANSACOutlierRejectionThreshold(DBL_MAX);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned);

  std::cout << (strrchr(argv[2], '/') + 1)
            << " converged:" << icp.hasConverged()
            << " mse:" << icp.getFitnessScore()
            << " rmse:" << std::sqrt(icp.getFitnessScore()) << std::endl;

  {
    std::ofstream stream(std::string() + argv[2] + ".alignment.xyz");
    for (auto &p : aligned.points) {
      stream << p.x << " " << p.y << " " << p.z << "\n";
    }
  }
}
