#include "../mc1/dag.hpp"

#include <iostream>
#include <span>
#include <vector>

using std::as_bytes, std::span, std::vector;
using std::cin, std::cout, std::istream, std::istreambuf_iterator, std::ostream;
using nlohmann::json;

namespace {

bool dag2json(istream& in, ostream& out)
{
  auto data = vector(istreambuf_iterator(in), istreambuf_iterator<char>());
  auto bytes = as_bytes(span(data));

  if (auto dag = MiniCollider::dag::parse(bytes)) {
    if (bytes.empty()) {
      out << json(dag.value());

      return true;
    }
  }

  return false;
}

}

int main()
{
  return dag2json(cin, cout)? EXIT_SUCCESS: EXIT_FAILURE;
}
