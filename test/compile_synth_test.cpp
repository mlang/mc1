#include <cstddef>
#include <format>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "compiler.hpp"
#include "dag.hpp"

namespace {

int run(mc1::DAG dag)
{
  constexpr unsigned int sample_rate = 44100;
  constexpr size_t block_size = 32;

  auto result = mc1::compile(dag, sample_rate, block_size);
  auto synth = result[dag.name];

  std::vector<float> controls = dag.controls;
  std::vector<float> abus(2 * block_size, 0.0f);

  std::cout << dag;
  std::cout << std::fixed << std::setprecision(6);

  for (int iter = 0; iter < 3; ++iter) {
    synth.process(controls.data(), abus.data());

    std::cout << "process call " << iter << "\n";
    std::cout << "i\tch0\tch1\n";
    for (size_t i = 0; i < block_size; ++i) {
      std::cout << i << '\t' << abus[i] << '\t' << abus[block_size + i] << "\n";
    }
    std::cout << "\n";
  }

  return 0;
}

}

int main()
{
  std::string input(
    std::istreambuf_iterator<char>(std::cin),
    std::istreambuf_iterator<char>()
  );
  if (input.empty()) {
    std::cerr << "No DAG bytes provided on stdin\n";
    return 2;
  }

  auto span = std::as_bytes(std::span{input});
  auto dag = mc1::DAG::parse(span);
  if (!dag || !span.empty()) {
    std::cerr << "Failed to parse DAG payload\n";
    return 2;
  }

  try {
    return run(std::move(*dag));
  } catch (const std::exception &e) {
    std::cerr << std::format("Compilation run failed: {}\n", e.what());
    return 1;
  }
}
