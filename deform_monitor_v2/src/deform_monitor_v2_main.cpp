#include "deform_monitor_v2/deform_monitor_v2_node.hpp"

#include <iostream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "deform_monitor_v2_node");

  try {
    deform_monitor_v2::DeformMonitorV2Node node;
    node.Run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[微动监测V2] 异常退出: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[微动监测V2] 未知异常退出" << std::endl;
  }
  return 1;
}
