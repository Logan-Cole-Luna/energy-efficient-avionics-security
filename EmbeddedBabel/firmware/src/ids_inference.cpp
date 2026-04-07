/*
 * IDS Inference Module Implementation
 * 
 * Decision tree inference with DWT timing for STM32F373
 *
 * Copyright (C) 2024
 */

#include "ids_inference.hpp"
#include "ids_model.h"
#include "ids_scaler.h"

#include <hal.h>
#include <cstring>

namespace ids
{

/* Stack canary region - 1 KB in .bss */
static std::uint32_t s_canary[CANARY_WORDS];

void initDWT()
{
    /* Enable DWT and cycle counter */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

std::uint32_t getDWTCycles()
{
    return DWT->CYCCNT;
}

void paintCanary()
{
    for (unsigned i = 0; i < CANARY_WORDS; i++) {
        s_canary[i] = CANARY_VALUE;
    }
    /* Compiler barrier - prevent paint from being optimized away */
    __asm volatile("" ::: "memory");
}

std::uint32_t getCanaryUsedBytes()
{
    unsigned i = 0;
    while (i < CANARY_WORDS && s_canary[i] == CANARY_VALUE) {
        i++;
    }
    return static_cast<std::uint32_t>(CANARY_WORDS - i) * 4;
}

void scaleFeatures(float* features)
{
    ids_scale_features(features);
}

int predict(const float* features)
{
    return ids_predict(features);
}

void runInference(float* features, std::uint8_t ground_truth, SampleResult& out_result)
{
    /* Scale features in-place */
    scaleFeatures(features);
    
    /* Isolated inference window - mask ALL interrupts */
    __disable_irq();
    __DSB();  /* Drain store buffer */
    __ISB();  /* Flush pipeline */
    
    std::uint32_t t0 = getDWTCycles();
    int prediction = predict(features);
    std::uint32_t elapsed = getDWTCycles() - t0;
    
    __enable_irq();
    /* End of isolated window */
    
    out_result.prediction = static_cast<std::uint8_t>(prediction);
    out_result.ground_truth = ground_truth;
    out_result.cycles = elapsed;
    out_result.stack_used_bytes = getCanaryUsedBytes();
    out_result.reserved0 = 0;
    out_result.reserved1 = 0;
}

void computeSummary(
    std::uint32_t n_samples,
    std::uint32_t tp, std::uint32_t tn, std::uint32_t fp, std::uint32_t fn,
    std::uint32_t cycles_min, std::uint32_t cycles_max,
    std::uint64_t cycles_sum, std::uint32_t stack_max,
    BenchSummary& out_summary)
{
    float n = static_cast<float>(n_samples);
    float accuracy  = static_cast<float>(tp + tn) / n;
    float precision = (tp + fp) > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    float recall    = (tp + fn) > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    float f1        = (precision + recall) > 0.0f
                      ? 2.0f * precision * recall / (precision + recall) : 0.0f;
    float fpr       = (fp + tn) > 0 ? static_cast<float>(fp) / (fp + tn) : 0.0f;
    
    float cycles_mean = static_cast<float>(cycles_sum / n_samples);
    float inf_us = cycles_mean / (HCLK_HZ / 1e6f);
    float energy_nj = (static_cast<float>(VDD_MV) / 1000.0f)
                    * (static_cast<float>(RUN_CURRENT_UA) / 1e6f)
                    * (inf_us / 1e6f) * 1e9f;
    
    out_summary.n_samples = n_samples;
    out_summary.n_correct = tp + tn;
    out_summary.tp = tp;
    out_summary.tn = tn;
    out_summary.fp = fp;
    out_summary.fn = fn;
    out_summary.cycles_min = cycles_min;
    out_summary.cycles_max = cycles_max;
    out_summary.cycles_sum_hi = static_cast<std::uint32_t>(cycles_sum >> 32);
    out_summary.cycles_sum_lo = static_cast<std::uint32_t>(cycles_sum & 0xFFFFFFFF);
    out_summary.stack_used_bytes_max = stack_max;
    out_summary.energy_per_inf_nj = energy_nj;
    out_summary.inference_us_mean = inf_us;
    out_summary.accuracy = accuracy;
    out_summary.precision = precision;
    out_summary.recall = recall;
    out_summary.f1 = f1;
    out_summary.fpr = fpr;
}

}  // namespace ids
