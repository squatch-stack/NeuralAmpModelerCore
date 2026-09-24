// Tests for ConvNet

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/convnet.h"

namespace test_convnet
{
// Test basic ConvNet construction and processing
void test_convnet_basic()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, !batchnorm=true, 1) -> 2*1*2 = 4 weights + 2 bias = 6 total
  // Block 1: Conv1D (2, 2, 2, !batchnorm=true, 2) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 6 + 10 + 3 = 19 weights
  std::vector<float> weights;
  // Block 0 weights (4 weights: kernel[0] and kernel[1], each 2x1) + 2 bias
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights: kernel[0] and kernel[1], each 2x2) + 2 bias
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights (2 weights + 1 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero and finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with batchnorm
void test_convnet_batchnorm()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = true;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 1, 2, !batchnorm=false, 1) -> 2*1*1 = 2 weights (no bias when batchnorm=true)
  // BatchNorm: running_mean(1) + running_var(1) + weight(1) + bias(1) + eps(1) = 5 weights
  // Head: (1, 1) weight + 1 bias = 2 weights
  // Total: 2 + 5 + 2 = 9 weights
  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1], no bias)
  weights.insert(weights.end(), {1.0f, 1.0f});
  // BatchNorm weights (5: mean, var, weight, bias, eps)
  weights.insert(weights.end(), {0.0f, 1.0f, 1.0f, 0.0f, 1e-5f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with multiple blocks
void test_convnet_multiple_blocks()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, !batchnorm=true, 1) -> 2*1*2 = 4 weights + 2 bias = 6 total
  // Block 1: Conv1D (2, 2, 2, !batchnorm=true, 2) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Block 2: Conv1D (2, 2, 2, !batchnorm=true, 4) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 6 + 10 + 10 + 3 = 29 weights
  std::vector<float> weights;
  // Block 0 weights (4 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 2 weights (8 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with zero input
void test_convnet_zero_input()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  convnet.Reset(expected_sample_rate, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with different buffer sizes
void test_convnet_different_buffer_sizes()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test with different buffer sizes
  convnet.Reset(expected_sample_rate, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  NAM_SAMPLE* inputPtrs1[] = {input1.data()};
  NAM_SAMPLE* outputPtrs1[] = {output1.data()};
  convnet.process(inputPtrs1, outputPtrs1, 32);

  convnet.Reset(expected_sample_rate, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  NAM_SAMPLE* inputPtrs2[] = {input2.data()};
  NAM_SAMPLE* outputPtrs2[] = {output2.data()};
  convnet.process(inputPtrs2, outputPtrs2, 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test ConvNet prewarm functionality
void test_convnet_prewarm()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (4 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 2 weights (8 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights (2 weights + 1 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test that prewarm can be called without errors
  convnet.Reset(expected_sample_rate, 64);
  convnet.prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  convnet.process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test multiple process() calls (ring buffer functionality)
void test_convnet_multiple_calls()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 2;
  convnet.Reset(expected_sample_rate, numFrames);

  // Multiple calls should work correctly with ring buffer
  for (int i = 0; i < 5; i++)
  {
    std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
    std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
    NAM_SAMPLE* inputPtrs[] = {input.data()};
    NAM_SAMPLE* outputPtrs[] = {output.data()};
    convnet.process(inputPtrs, outputPtrs, numFrames);

    // Output should be finite
    for (int j = 0; j < numFrames; j++)
    {
      assert(std::isfinite(output[j]));
    }
  }
}
// A call with more frames than the maximum buffer size must give exactly what the host gets by making calls of at
// most that size. Each block's Conv1D holds only that many frames, so such a call used to write past its buffers.
void test_convnet_oversized_call_matches_consecutive_calls()
{
  const int in_channels = 2;
  const int out_channels = 2;
  const int channels = 3;
  const std::vector<int> dilations{1, 2, 4};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
  const double sample_rate = 48000.0;
  const int max_buffer_size = 32;
  const int call_size = 100; // Not a multiple of max_buffer_size, so each call's last chunk is short
  const int num_calls = 20;

  // Blocks: kernel size 2 with bias; head: channels -> out_channels with bias
  const int num_weights = (in_channels * channels * 2 + channels) + 2 * (channels * channels * 2 + channels)
                          + (channels * out_channels + out_channels);
  std::vector<float> weights(num_weights);
  for (int i = 0; i < num_weights; i++)
  {
    weights[i] = 0.5f * std::sin(0.7f * (float)i);
  }
  std::vector<float> weights_copy = weights;
  nam::convnet::ConvNet consecutive(
    in_channels, out_channels, channels, dilations, false, activation, weights, sample_rate);
  nam::convnet::ConvNet oversized(
    in_channels, out_channels, channels, dilations, false, activation, weights_copy, sample_rate);
  consecutive.Reset(sample_rate, max_buffer_size);
  oversized.Reset(sample_rate, max_buffer_size);

  std::vector<std::vector<NAM_SAMPLE>> input(in_channels, std::vector<NAM_SAMPLE>(call_size));
  std::vector<std::vector<NAM_SAMPLE>> expected(out_channels, std::vector<NAM_SAMPLE>(call_size));
  std::vector<std::vector<NAM_SAMPLE>> actual(out_channels, std::vector<NAM_SAMPLE>(call_size));
  std::vector<NAM_SAMPLE*> input_ptrs(in_channels);
  std::vector<NAM_SAMPLE*> output_ptrs(out_channels);
  for (int call = 0; call < num_calls; call++)
  {
    for (int ch = 0; ch < in_channels; ch++)
    {
      for (int i = 0; i < call_size; i++)
      {
        const double t = (double)(call * call_size + i);
        input[ch][i] = (NAM_SAMPLE)(0.5 * std::sin(0.01 * t * (ch + 1)) * std::sin(0.37 * t));
      }
    }
    for (int start = 0; start < call_size; start += max_buffer_size)
    {
      for (int ch = 0; ch < in_channels; ch++)
      {
        input_ptrs[ch] = input[ch].data() + start;
      }
      for (int ch = 0; ch < out_channels; ch++)
      {
        output_ptrs[ch] = expected[ch].data() + start;
      }
      consecutive.process(input_ptrs.data(), output_ptrs.data(), std::min(max_buffer_size, call_size - start));
    }
    for (int ch = 0; ch < in_channels; ch++)
    {
      input_ptrs[ch] = input[ch].data();
    }
    for (int ch = 0; ch < out_channels; ch++)
    {
      output_ptrs[ch] = actual[ch].data();
    }
    oversized.process(input_ptrs.data(), output_ptrs.data(), call_size);
    assert(actual == expected);
  }
}
}; // namespace test_convnet
