#pragma once

#include <string>
#include <memory>
#include "fbx_model.hpp"

namespace marionette::preprocess
{
    std::shared_ptr<node_t> load_model(const std::string &filename);
}
