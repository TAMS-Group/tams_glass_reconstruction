#include "common.h"

#include <pcl/point_types.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

int main(int argc, char **argv) {

  static const double grid_resolution = 0.005;
  static const double mesh_threshold = 0.02;

  Eigen::Vector3d grid_center(0.918642, 0.704307, 0.708361);

  ros::init(argc, argv, "tams_glass_remesh", 0);

  std::ifstream voxel_file(argv[1]);

  if (!voxel_file) {
    throw std::runtime_error("failed to open voxel file");
  }

  struct {
    uint32_t size_x = 0, size_y = 1, size_z = 2, reserved = 0;
  } voxel_header;

  voxel_file.read((char *)&voxel_header, sizeof(voxel_header));

  std::cout << "grid size " << voxel_header.size_x << " " << voxel_header.size_y
            << " " << voxel_header.size_z << std::endl;

  if (voxel_header.size_x != voxel_header.size_y ||
      voxel_header.size_y != voxel_header.size_z) {
    throw std::runtime_error("currently only voxel cubes are supported");
  }

  VoxelGrid voxel_grid(grid_center, voxel_header.size_x, grid_resolution);

  for (size_t iz = 0; iz < voxel_grid.size(); iz++) {
    for (size_t iy = 0; iy < voxel_grid.size(); iy++) {
      for (size_t ix = 0; ix < voxel_grid.size(); ix++) {
        float v = 0.0f;
        voxel_file.read((char *)&v, sizeof(v));
        voxel_grid.at(ix, iy, iz) = v;
      }
    }
  }

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

              grid_[x * res_y_ * res_z_ + y * res_z_ + z] = mesh_threshold - a;

            } else {
              grid_[x * res_y_ * res_z_ + y * res_z_ + z] = 1.0;
            }
          }
        }
      }
    }
  };

  MatchingCubesVoxelGrid voxelizer(voxel_grid);
  voxelizer.setGridResolution(voxel_grid.size(), voxel_grid.size(),
                              voxel_grid.size());
  voxelizer.setInputCloud(cloud);
  voxelizer.setPercentageExtendGrid(0.0);
  pcl::PointCloud<pcl::PointNormal> mesh_vertices;
  std::vector<pcl::Vertices> mesh_polygons;
  voxelizer.reconstruct(mesh_vertices, mesh_polygons);

  {
    std::ofstream stream(std::string() + argv[1] + ".mesh.obj");
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
