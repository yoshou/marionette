#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace marionette::data_source {

bool encode_base64(const uint8_t *src, size_t size, std::string &dst);
bool decode_base64(const std::string &src, uint8_t *dst, size_t size, size_t *consumed);

}  // namespace marionette::data_source
