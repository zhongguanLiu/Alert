# FAST-LIO backend for ALERT

This directory contains the FAST-LIO backend used by ALERT to provide registered point clouds, odometry, and pose covariance.

The public tree is intentionally limited to the Livox Mid360 configurations used by ALERT:

- `launch/mapping_mid360.launch` with `config/mid360.yaml` for simulation
- `launch/mapping_mid360_real.launch` with `config/mid360_real.yaml` for real sensors

`livox_ros_driver` is an external build and runtime dependency. The backend is derived from [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO); its upstream license is retained in `LICENSE`.
