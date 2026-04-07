/*
 * IDS Feature Scaler
 * StandardScaler parameters for 15 CAN features
 * Auto-exported from phase4b_nsl_can_scaler.joblib
 * 
 * Feature order (must match host_inference_runner.py FEATURE_NAMES):
 *   0: can_id_norm
 *   1: dlc
 *   2: data_mean
 *   3: data_std
 *   4: data_entropy
 *   5: data_range
 *   6: hamming_dist
 *   7: inter_arrival_mean
 *   8: inter_arrival_std
 *   9: id_freq
 *  10: bus_load
 *  11: unique_ids
 *  12: dlc_anomaly
 *  13: id_is_known
 *  14: payload_delta
 *
 * Target: STM32F373 (Cortex-M4F, 72 MHz)
 */
#pragma once

#include <stdint.h>

#define IDS_N_FEATURES 15

/* Feature index constants */
#define IDS_FEAT_CAN_ID_NORM         0
#define IDS_FEAT_DLC                 1
#define IDS_FEAT_DATA_MEAN           2
#define IDS_FEAT_DATA_STD            3
#define IDS_FEAT_DATA_ENTROPY        4
#define IDS_FEAT_DATA_RANGE          5
#define IDS_FEAT_HAMMING_DIST        6
#define IDS_FEAT_INTER_ARRIVAL_MEAN  7
#define IDS_FEAT_INTER_ARRIVAL_STD   8
#define IDS_FEAT_ID_FREQ             9
#define IDS_FEAT_BUS_LOAD            10
#define IDS_FEAT_UNIQUE_IDS          11
#define IDS_FEAT_DLC_ANOMALY         12
#define IDS_FEAT_ID_IS_KNOWN         13
#define IDS_FEAT_PAYLOAD_DELTA       14

/*
 * StandardScaler parameters (fit on training data)
 * Mean and scale arrays for z-score normalization: x_scaled = (x - mean) / scale
 * 
 * Extracted from trained model for 15-feature CAN IDS.
 * Order matches FEATURE_NAMES in host_inference_runner.py.
 */
static const float ids_scaler_mean[IDS_N_FEATURES] = {
    0.376404f,    /* can_id_norm */
    6.833333f,    /* dlc */
    35.634546f,   /* data_mean */
    56.605397f,   /* data_std */
    0.955393f,    /* data_entropy */
    144.781583f,  /* data_range */
    10.190433f,   /* hamming_dist */
    0.007000f,    /* inter_arrival_mean */
    0.000000f,    /* inter_arrival_std */
    157.452096f,  /* id_freq */
    874.693565f,  /* bus_load */
    5.999917f,    /* unique_ids */
    0.000000f,    /* dlc_anomaly */
    1.000000f,    /* id_is_known */
    276.030872f   /* payload_delta */
};

static const float ids_scaler_scale[IDS_N_FEATURES] = {
    0.000834f,    /* can_id_norm */
    2.608746f,    /* dlc */
    33.157359f,   /* data_mean */
    46.228392f,   /* data_std */
    0.720550f,    /* data_entropy */
    114.589384f,  /* data_range */
    10.869287f,   /* hamming_dist */
    0.000040f,    /* inter_arrival_mean */
    0.000000f,    /* inter_arrival_std - CAUTION: zero scale, feature is constant */
    2.494435f,    /* id_freq */
    6.964801f,    /* bus_load */
    0.017480f,    /* unique_ids */
    1.000000f,    /* dlc_anomaly */
    1.000000f,    /* id_is_known */
    344.548900f   /* payload_delta */
};

/*
 * Apply StandardScaler in-place to a raw feature vector
 * @param x  Raw feature vector (15 floats), modified in-place
 */
static inline void ids_scale_features(float *x) {
    for (int i = 0; i < IDS_N_FEATURES; i++) {
        /* Handle zero scale (constant feature) - result is 0 */
        if (ids_scaler_scale[i] > 1e-10f) {
            x[i] = (x[i] - ids_scaler_mean[i]) / ids_scaler_scale[i];
        } else {
            x[i] = 0.0f;
        }
    }
}
