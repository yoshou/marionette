#pragma once

#include <string>
#include <map>

namespace marionette::preprocess
{
    /**
     * Convert T-pose FBX model to JSON format
     * @param input_path Path to input FBX file
     * @param output_path Path to output JSON file
     * @param scale Scale factor to apply to the model (default: 0.01)
     * @return 0 on success, non-zero on error
     */
    int convert_tpose(const std::string& input_path, const std::string& output_path, float scale = 0.01f);

    /**
     * Convert tracking model FBX to JSON format
     * @param input_path Path to input FBX file
     * @param output_path Path to output JSON file
     * @param scale Scale factor to apply to the model (default: 0.01)
     * @return 0 on success, non-zero on error
     */
    int convert_tracking_model(const std::string& input_path, const std::string& output_path, float scale = 0.01f);
}
