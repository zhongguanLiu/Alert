/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/deform_monitor_v2_node.hpp"

#include <iostream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "deform_monitor_v2_node");

  try {
    deform_monitor_v2::DeformMonitorV2Node node;
    node.Run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[deform_monitor_v2] Fatal error: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[deform_monitor_v2] Unknown fatal error" << std::endl;
  }
  return 1;
}
