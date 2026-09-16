#pragma once

#include "execution.hpp"

#include <memory>
#include <string>

namespace trident {

using ExchangeFactory = std::unique_ptr<IntraNodeExchange> (*)(const ProcessGrid&, const ExecutionPlan&);
int runBenchmark(int argc, char** argv, const std::string& implementation, ExchangeFactory factory);

}  // namespace trident
