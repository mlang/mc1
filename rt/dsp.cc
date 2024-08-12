#include <libgccjit++.h>

#include <fstream>

bool read_dag(std::string fn) {
  std::ifstream file(fn, std::ios::binary);
  if (!file) return false;
  int32_t magic;
  if (!file.read(reinterpret_cast<char *>(&magic), sizeof(magic))) return false;
  if (magic != 1735287116) return false;
  int32_t clen;
  if (!file.read(reinterpret_cast<char *>(&clen), sizeof(clen))) return false;
  std::vector<float> constants;
  for (int i = 0; i != clen; i++) {
    float v;
    if (!file.read(reinterpret_cast<char *>(&v), sizeof(v))) return false;
    constants.push_back(v);
  }
  return true;
}

#include <iostream>
#include <utility>
#include <boost/asio.hpp>


using boost::asio::ip::udp;

class server {
  udp::socket socket;
  udp::endpoint sender_endpoint_;
  enum { max_length = 1024 };
  char data_[max_length];

public:
  server(boost::asio::io_context& io_context, short port)
  : socket{io_context, udp::endpoint(udp::v4(), port)}
  { do_receive(); }

private:
  void do_receive() {
    socket.async_receive_from(
      boost::asio::buffer(data_), sender_endpoint_,
      [this](boost::system::error_code ec, std::size_t bytes_recvd) {
        if (!ec && bytes_recvd > 0) {
          std::cout << bytes_recvd << " bytes received" << std::endl;
	  std::cout << *reinterpret_cast<int *>(&data_[0]) << std::endl;
          do_receive();
        }
      }
    );
  }

  void do_send(std::size_t length) {
    socket.async_send_to(
      boost::asio::buffer(data_, length), sender_endpoint_,
      [this](boost::system::error_code /*ec*/, std::size_t /*bytes_sent*/) {
        do_receive();
      }
    );
  }
};

int main(int argc, const char *argv[]) {
  boost::asio::io_context io;
  server srv(io, std::atoi(argv[1]));
  io.run();

  return 0;
}
