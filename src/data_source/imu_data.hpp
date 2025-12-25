#pragma once

#include <cstdint>

#define NUM_SENSORS (6)

namespace marionette::data_source {

#pragma pack(push, 1)
struct vec_3 {
  double x, y, z;
};
struct quat {
  double x, y, z, w;
};

struct bno055_sensor_info {
  char name[12];
  int32_t version;
  int32_t sensor_id;
  int32_t type;
  float max_value;
  float min_value;
  float resolution;
  int32_t min_delay;
};
struct bno055_imu_data {
  uint8_t system, gyro, accel, mag;
#if 0
    vec_3 orientation;
    vec_3 ang_velocity;
    vec_3 linear_accel;
    vec_3 magnetometer;
    vec_3 accelerometer;
    vec_3 gravity;
#endif
  quat orientation_quat;
};
struct bno055_calib_data {
  int16_t accel_offset_x;
  int16_t accel_offset_y;
  int16_t accel_offset_z;

  int16_t mag_offset_x;
  int16_t mag_offset_y;
  int16_t mag_offset_z;

  int16_t gyro_offset_x;
  int16_t gyro_offset_y;
  int16_t gyro_offset_z;
  int16_t accel_radius;
  int16_t mag_radius;
};
struct bno055_system_status {
  uint8_t system_status, self_test_results, system_error;
};
struct frame_data {
  uint64_t timestamp;
  bno055_imu_data imu[NUM_SENSORS];
#if 0
    int8_t temperature[NUM_SENSORS];
#endif
};
struct response_header {
  uint8_t id;
  uint8_t type;
  uint8_t code;
};
#pragma pack(pop)

}  // namespace marionette::data_source
