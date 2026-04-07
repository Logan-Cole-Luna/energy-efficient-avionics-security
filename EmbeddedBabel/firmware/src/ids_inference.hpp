/*
 * IDS Inference Module
 * 
 * Provides decision tree inference with timing measurements
 * for CAN-based intrusion detection on STM32F373.
 *
 * Copyright (C) 2024
 */

#pragma once

#include <cstdint>

namespace ids
{

/* Hardware constants for energy estimation */
constexpr std::uint32_t HCLK_HZ         = 72000000UL;  /* STM32F373 max clock */
constexpr std::uint32_t VDD_MV          = 3300U;
constexpr std::uint32_t RUN_CURRENT_UA  = 27000U;      /* ~27 mA @ 72 MHz */

constexpr unsigned N_FEATURES = 15;
constexpr unsigned MAX_SAMPLES = 65535;  /* Max uint16 - samples processed one at a time */

/* Stack canary configuration */
constexpr std::uint32_t CANARY_VALUE = 0xDEADBEEFU;
constexpr unsigned CANARY_WORDS = 256;  /* 1 KB */

/**
 * Wire protocol structures - must match host_inference_runner.py exactly
 */
struct __attribute__((packed)) SampleResult {
    std::uint8_t  prediction;
    std::uint8_t  ground_truth;
    std::uint32_t cycles;
    std::uint32_t stack_used_bytes;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
};
static_assert(sizeof(SampleResult) == 18, "SampleResult must be 18 bytes");

struct __attribute__((packed)) BenchSummary {
    std::uint32_t n_samples;
    std::uint32_t n_correct;
    std::uint32_t tp, tn, fp, fn;
    std::uint32_t cycles_min;
    std::uint32_t cycles_max;
    std::uint32_t cycles_sum_hi;
    std::uint32_t cycles_sum_lo;
    std::uint32_t stack_used_bytes_max;
    float         energy_per_inf_nj;
    float         inference_us_mean;
    float         accuracy;
    float         precision;
    float         recall;
    float         f1;
    float         fpr;
};
static_assert(sizeof(BenchSummary) == 72, "BenchSummary must be 72 bytes");

/**
 * Initialize DWT cycle counter for timing measurements
 */
void initDWT();

/**
 * Get current DWT cycle count
 */
std::uint32_t getDWTCycles();

/**
 * Paint stack canary region before benchmark
 */
void paintCanary();

/**
 * Get number of bytes used by stack (canary words overwritten)
 */
std::uint32_t getCanaryUsedBytes();

/**
 * Scale features in-place using StandardScaler parameters
 * @param features  14-element float array, modified in-place
 */
void scaleFeatures(float* features);

/**
 * Run decision tree inference on scaled features
 * @param features  14-element scaled float array
 * @return 0 = normal, 1 = attack
 */
int predict(const float* features);

/**
 * Isolated inference with timing measurement
 * Disables interrupts during prediction for accurate cycle count.
 * 
 * @param features      Raw features (14 floats), will be scaled in-place
 * @param ground_truth  Expected label for statistics
 * @param out_result    Result structure to fill
 */
void runInference(float* features, std::uint8_t ground_truth, SampleResult& out_result);

/**
 * Compute summary statistics from sample results
 * 
 * @param n_samples     Total number of samples processed
 * @param tp,tn,fp,fn   Confusion matrix values
 * @param cycles_min    Minimum cycles observed
 * @param cycles_max    Maximum cycles observed
 * @param cycles_sum    Sum of all cycles (64-bit)
 * @param stack_max     Maximum stack usage observed
 * @param out_summary   Summary structure to fill
 */
void computeSummary(
    std::uint32_t n_samples,
    std::uint32_t tp, std::uint32_t tn, std::uint32_t fp, std::uint32_t fn,
    std::uint32_t cycles_min, std::uint32_t cycles_max,
    std::uint64_t cycles_sum, std::uint32_t stack_max,
    BenchSummary& out_summary
);

}  // namespace ids
