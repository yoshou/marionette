
#include <iostream>
#include <model_converter.hpp>

int main(int argc, const char *argv[]) {
  // Convert both JSON files
  std::cout << "Converting DefaultPose.json..." << std::endl;
  if (marionette::preprocess::convert_tpose("../data/FullTrackingBone.fbx",
                                            "../data/DefaultPose.json") != 0) {
    std::cerr << "Failed to convert DefaultPose.json" << std::endl;
    return 1;
  }
  std::cout << "Successfully converted." << std::endl;

  std::cout << "\nConverting TrackingModel.json..." << std::endl;
  if (marionette::preprocess::convert_tracking_model("../data/TrackingModel.fbx",
                                                     "../data/TrackingModel.json") != 0) {
    std::cerr << "Failed to convert TrackingModel.json" << std::endl;
    return 1;
  }
  std::cout << "Successfully converted." << std::endl;

  std::cout << "\nAll conversions completed successfully." << std::endl;

  return 0;
}
