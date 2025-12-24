#pragma once

#include <memory>
#include <string>
#include <vector>

namespace marionette::data_source {

class serial_port {
  struct handle_t;

  std::unique_ptr<handle_t> handle;

 public:
  serial_port();
  virtual ~serial_port();

  static std::vector<std::string> get_serial_port_names();

  void open(std::string name);
  void close();
  void set_baudrate(uint32_t baudrate);

  size_t get_received_size() const;
  size_t read(uint8_t* buf, size_t size);
  void clear();
};

}  // namespace marionette::data_source
