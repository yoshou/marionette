#pragma once

#include <memory>
#include <string>

#include "fbx_model.hpp"

namespace marionette::preprocess {
std::shared_ptr<node_t> load_model(const std::string &filename);
}
