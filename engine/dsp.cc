#include <libgccjit++.h>

#include <fstream>

#include <iostream>
#include <utility>
#include <boost/asio.hpp>

using boost::asio::ip::udp;

class server {
  udp::socket socket;
  udp::endpoint sender;
  char data[1024];

public:
  server(boost::asio::io_context& io_context, short port)
  : socket{io_context, udp::endpoint(udp::v4(), port)}
  { do_receive(); }

private:
  void do_receive() {
    socket.async_receive_from(
      boost::asio::buffer(data), sender,
      [this](boost::system::error_code ec, std::size_t bytes_recvd) {
        if (!ec && bytes_recvd > 0) {
          std::cout << bytes_recvd << " bytes received" << std::endl;
	  std::cout << *reinterpret_cast<int *>(&data[0]) << std::endl;
          do_receive();
        }
      }
    );
  }

  void do_send(std::size_t length) {
    socket.async_send_to(
      boost::asio::buffer(data, length), sender,
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
