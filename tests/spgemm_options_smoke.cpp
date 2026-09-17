#include "spgemm_common.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

Options parse(const std::string& option, const std::string& value) {
    std::vector<std::string> arguments{
        "options_test", "--nnz-per-row", "1", "--b-nnz-per-row", "1", option, value};
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    const int argc = static_cast<int>(argv.size());
    argv.push_back(nullptr);
    return parseOptions(argc, argv.data(), "options_test", "unused.tsv");
}

void expectRejected(const std::string& option, const std::string& value) {
    try {
        parse(option, value);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(option) == std::string::npos) {
            throw std::runtime_error("error does not identify " + option);
        }
        return;
    }
    throw std::runtime_error("accepted invalid value '" + value + "' for " + option);
}

}  // namespace

int main() {
    try {
        const std::vector<std::pair<std::string, int Options::*>> positiveOptions{
            {"--rows", &Options::rows}, {"--cols", &Options::cols},
            {"--b-cols", &Options::bCols}, {"--nnz-per-row", &Options::nonZerosPerRow},
            {"--b-nnz-per-row", &Options::bNonZerosPerRow}, {"--threads", &Options::threads},
            {"--repeats", &Options::repeats}, {"--iterations", &Options::repeats},
            {"--trials", &Options::trials}, {"--chunk", &Options::chunk}};
        const std::vector<std::string> malformed{
            "", "junk", "1e3", "2junk", "2.5", "0x10", "2 ", "2\t", "2\n",
            "2147483648", "999999999999999999999999", "-1"};
        for (const auto& [option, member] : positiveOptions) {
            for (const std::string value : {"2", "02", "+2"}) {
                if (parse(option, value).*member != 2) {
                    throw std::runtime_error("wrong parsed value for " + option);
                }
            }
            expectRejected(option, "0");
            for (const auto& value : malformed) expectRejected(option, value);
        }
        for (const auto& value : malformed) expectRejected("--warmup", value);
        expectRejected("--warmup", "0suffix");
        if (parse("--warmup", "0").warmup != 0 || parse("--warmup", "2").warmup != 2) {
            throw std::runtime_error("warmup must accept zero and positive integers");
        }
        const int max = std::numeric_limits<int>::max();
        if (parse("--rows", std::to_string(max)).rows != max ||
            parse("--warmup", std::to_string(max)).warmup != max) {
            throw std::runtime_error("parser rejected the signed-int upper bound");
        }
        std::cout << "Numeric options: valid integers, malformed input and bounds: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
