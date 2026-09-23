// Tests that ConfigParserRegistry registers Core's own architectures itself (issue #327)
//
// Each architecture used to register itself from a static object in its own translation unit. A static-library link
// drops those units when nothing references them, and until they had been initialized the registry was empty.

#include <cassert>
#include <memory>
#include <stdexcept>

#include "NAM/model_config.h"

namespace test_builtin_registration
{
// Read during static initialization. run_tests.cpp is linked first, so this runs before the translation units that
// define the architectures have been initialized.
static const bool kWaveNetRegisteredDuringStaticInit = nam::ConfigParserRegistry::instance().has("WaveNet");

void test_builtins_registered()
{
  auto& registry = nam::ConfigParserRegistry::instance();
  for (const char* name : {"ConvNet", "Linear", "LSTM", "Sequential", "SlimmableContainer", "WaveNet"})
  {
    assert(registry.has(name));
  }
  assert(kWaveNetRegisteredDuringStaticInit);
}

void test_builtin_cannot_be_registered_again()
{
  bool threw = false;
  try
  {
    nam::ConfigParserRegistry::instance().registerParser(
      "WaveNet", [](const nlohmann::json&, double) -> std::unique_ptr<nam::ModelConfig> { return nullptr; });
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

} // namespace test_builtin_registration
