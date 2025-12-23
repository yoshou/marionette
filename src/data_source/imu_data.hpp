#pragma once

#include <cstdint>

#define NUM_SENSORS (6)

#pragma pack(push, 1)
struct Vec3
{
    double x, y, z;
};
struct Quat
{
    double x, y, z, w;
};

struct BNO055SensorInfo
{
    char name[12];
    int32_t version;
    int32_t sensor_id;
    int32_t type;
    float max_value;
    float min_value;
    float resolution;
    int32_t min_delay;
};
struct BNO055ImuData
{
    uint8_t system, gyro, accel, mag;
#if 0
    Vec3 orientation;
    Vec3 ang_velocity;
    Vec3 linear_accel;
    Vec3 magnetometer;
    Vec3 accelerometer;
    Vec3 gravity;
#endif
    Quat orientation_quat;
};
struct BNO055CalibData
{
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
struct BNO055SystemStatus
{
    uint8_t system_status, self_test_results, system_error;
};
struct FrameData
{
    uint64_t timestamp;
    BNO055ImuData imu[NUM_SENSORS];
#if 0
    int8_t temperature[NUM_SENSORS];
#endif
};
struct ResponseHeader
{
    uint8_t id;
    uint8_t type;
    uint8_t code;
};
#pragma pack(pop)
