#include "serial_port.hpp"

#ifdef _WIN32

// clang-format off
#include <windows.h>
#include <setupapi.h>
// clang-format on

#include <sstream>

#pragma comment(lib, "setupapi.lib")

#define PATH "\\\\.\\"
#define PORTS "Ports"
#define PORTNAME "PortName"

namespace marionette::data_source {

std::vector<std::string> serial_port::get_serial_port_names() {
  std::vector<std::string> list;
  HDEVINFO hinfo = NULL;
  SP_DEVINFO_DATA info_data = {0};
  info_data.cbSize = sizeof(SP_DEVINFO_DATA);

  GUID guid;
  unsigned long guid_size = 0;
  if (SetupDiClassGuidsFromName(PORTS, &guid, 1, &guid_size) == FALSE) return list;

  hinfo = SetupDiGetClassDevs(&guid, 0, 0, DIGCF_PRESENT | DIGCF_PROFILE);
  if (hinfo == INVALID_HANDLE_VALUE) return list;

  char buff[MAX_PATH];
  std::string name;
  std::string fullname;
  unsigned int index = 0;
  while (SetupDiEnumDeviceInfo(hinfo, index, &info_data)) {
    unsigned long type;
    unsigned long size;

    if (SetupDiGetDeviceRegistryProperty(hinfo, &info_data, SPDRP_FRIENDLYNAME, &type, (PBYTE)buff,
                                         MAX_PATH, &size) == TRUE) {
      fullname = buff;
    } else if (SetupDiGetDeviceRegistryProperty(hinfo, &info_data, SPDRP_DEVICEDESC, &type,
                                                (PBYTE)buff, MAX_PATH, &size) == TRUE) {
      fullname = buff;
    }

    HKEY hkey = SetupDiOpenDevRegKey(hinfo, &info_data, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);
    if (hkey) {
      RegQueryValueEx(hkey, PORTNAME, 0, &type, (LPBYTE)buff, &size);
      RegCloseKey(hkey);
      name = buff;
    }
    list.push_back(name);
    index++;
  }
  SetupDiDestroyDeviceInfoList(hinfo);
  return list;
}

struct serial_port::handle_t {
  HANDLE value;
};

serial_port::serial_port() : handle(std::make_unique<handle_t>()) {}

serial_port::~serial_port() { close(); }

void serial_port::open(std::string name) {
  handle->value = CreateFileA((std::string(PATH) + name).c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (handle->value == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("Failed to open com port: " + name);
  }

  DCB dcb;
  GetCommState(handle->value, &dcb);
  dcb.BaudRate = 115200;
  dcb.ByteSize = 8;
  dcb.Parity = NOPARITY;
  dcb.StopBits = ONESTOPBIT;
  dcb.fOutxCtsFlow = FALSE;
  dcb.fRtsControl = RTS_CONTROL_DISABLE;
  SetCommState(handle->value, &dcb);

  uint32_t read_buf_size = 1024;
  uint32_t write_buf_size = 1024;
  SetupComm(handle->value, read_buf_size, write_buf_size);
}

void serial_port::close() {
  if (handle->value != INVALID_HANDLE_VALUE) {
    CloseHandle(handle->value);
  }
}

void serial_port::set_baudrate(uint32_t baudrate) {
  DCB dcb;
  GetCommState(handle->value, &dcb);
  dcb.BaudRate = baudrate;
  SetCommState(handle->value, &dcb);
}

size_t serial_port::get_received_size() const {
  DWORD errors;
  COMSTAT stat;
  ClearCommError(handle->value, &errors, &stat);
  return stat.cbInQue;
}

size_t serial_port::read(uint8_t *buf, size_t size) {
  unsigned long read_len;
  ReadFile(handle->value, buf, size, &read_len, NULL);
  return read_len;
}

void serial_port::clear() { PurgeComm(handle->value, PURGE_RXCLEAR | PURGE_TXCLEAR); }

}  // namespace marionette::data_source

#elif defined(__linux__)

#include <asm/ioctls.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace marionette::data_source {

static const std::string get_driver(const fs::path &dir) {
  const auto driver_dir = dir / "device/driver";
  if (!exists(driver_dir)) {
    return std::string();
  }
  return fs::read_symlink(driver_dir).filename().generic_string();
}

static bool is_console(const std::string &driver) { return driver == "serial8250"; }

std::vector<std::string> serial_port::get_serial_port_names() {
  std::vector<std::string> list;
  fs::path p("/sys/class/tty/");
  try {
    if (!exists(p)) {
      throw std::runtime_error(p.generic_string() + " does not exist");
    } else {
      for (const auto &x : fs::directory_iterator(p)) {
        const auto filename = x.path().filename().generic_string();
        if (filename == ".." || filename == ".") {
          continue;
        }
        if (is_symlink(x.symlink_status())) {
          fs::path symlink_points_at = read_symlink(x);
          fs::path canonical_path = fs::canonical(p / symlink_points_at);
          const auto driver = get_driver(canonical_path);
          if (!driver.empty() && !is_console(driver)) {
            list.push_back((fs::path("/dev") / filename).generic_string());
          }
        }
      }
    }
  } catch (const fs::filesystem_error &ex) {
    throw ex;
  }
  std::sort(list.begin(), list.end());
  return list;
}

struct serial_port::handle_t {
  int fd = -1;
  termios tms;
};

serial_port::serial_port() : handle(std::make_unique<handle_t>()) {}

serial_port::~serial_port() { close(); }

void serial_port::open(std::string name) {
  const auto baudrate = B9600;
  handle->fd = ::open(name.c_str(), O_RDWR | O_NOCTTY);
  if (handle->fd < 0) {
    throw std::runtime_error("Failed to open com port: " + name);
  }

  tcgetattr(handle->fd, &handle->tms);

  termios tms = handle->tms;
  tms.c_iflag = 0;
  tms.c_oflag = 0;
  tms.c_lflag = 0;
  tms.c_cc[VMIN] = 1;
  tms.c_cc[VTIME] = 0;

  cfmakeraw(&tms);

  // BaudRate
  cfsetispeed(&tms, baudrate);
  cfsetospeed(&tms, baudrate);

  // Parity
  tms.c_cflag &= ~PARENB;
  tms.c_cflag &= ~PARODD;

  // DataBit
  tms.c_cflag &= ~CSIZE;
  tms.c_cflag |= CS8;

  // StopBit
  tms.c_cflag &= ~CSTOPB;

  // Configure flow control
  tms.c_cflag &= ~CRTSCTS;

  tms.c_cflag |= CREAD | CLOCAL;

  tcsetattr(handle->fd, TCSANOW, &tms);
  clear();
}

void serial_port::close() {
  if (handle->fd >= 0) {
    tcsetattr(handle->fd, TCSANOW, &handle->tms);
    ::close(handle->fd);
    handle->fd = -1;
  }
}

static speed_t get_speed(uint32_t baudrate) {
  switch (baudrate) {
    case 0:
      return B0;
    case 50:
      return B50;
    case 75:
      return B75;
    case 110:
      return B110;
    case 134:
      return B134;
    case 150:
      return B150;
    case 200:
      return B200;
    case 300:
      return B300;
    case 600:
      return B600;
    case 1200:
      return B1200;
    case 1800:
      return B1800;
    case 2400:
      return B2400;
    case 4800:
      return B4800;
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    case 460800:
      return B460800;
    case 500000:
      return B500000;
    case 576000:
      return B576000;
    case 921600:
      return B921600;
    case 1000000:
      return B1000000;
    case 1152000:
      return B1152000;
    case 1500000:
      return B1500000;
    case 2000000:
      return B2000000;
    case 2500000:
      return B2500000;
    case 3000000:
      return B3000000;
    case 3500000:
      return B3500000;
    case 4000000:
      return B4000000;
    default:
      throw std::runtime_error("Not supporting baudrate: " + std::to_string(baudrate));
  }
}

void serial_port::set_baudrate(uint32_t baudrate) {
  if (handle->fd < 0) {
    throw std::runtime_error("Port not opened");
  }
  termios tms;
  tcgetattr(handle->fd, &tms);
  cfsetispeed(&tms, get_speed(baudrate));
  cfsetospeed(&tms, get_speed(baudrate));
  tcsetattr(handle->fd, TCSANOW, &tms);
}

size_t serial_port::get_received_size() const {
  if (handle->fd < 0) {
    throw std::runtime_error("Port not opened");
  }
  int bytes = 0;
  if (ioctl(handle->fd, FIONREAD, &bytes) < 0) {
    throw std::runtime_error("Port control error");
  }
  return bytes;
}

size_t serial_port::read(uint8_t *buf, size_t size) {
  if (handle->fd < 0) {
    throw std::runtime_error("Port not opened");
  }
  const auto read_len = ::read(handle->fd, buf, size);
  if (read_len < 0) {
    throw std::runtime_error("Port read error");
  }
  return read_len;
}

void serial_port::clear() {
  if (handle->fd < 0) {
    throw std::runtime_error("Port not opened");
  }
  tcflush(handle->fd, TCIOFLUSH);
}

}  // namespace marionette::data_source

#else
#error "Not supported"
#endif
