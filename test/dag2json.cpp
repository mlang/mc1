#include "../mc1/dag.hpp"

#include <iostream>
#include <span>
#include <vector>

int main()
{
  auto data = std::vector(std::istreambuf_iterator<char>(std::cin),
                          std::istreambuf_iterator<char>());
  auto bytes = std::as_bytes(std::span(data));

  if (auto dag = MiniCollider::dag::parse(bytes)) {
    std::cout << nlohmann::json(dag.value());

    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
