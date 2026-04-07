/*
 * IDS Decision Tree Model
 * Auto-exported from phase4b_nsl_can_decision_tree.joblib
 * 
 * TinyDecisionTree for CAN IDS - 63 nodes, 15 features
 * Target: STM32F373 (Cortex-M4F, 72 MHz)
 */
#pragma once

#include <stdint.h>

#define IDS_N_NODES    63
#define IDS_N_FEATURES 15

/* Tree node feature indices (-2 means leaf node) */
static const int16_t ids_feature[IDS_N_NODES] = {
    2, 3, 0, 3, 3, -2, -2, 4, -2, -2, 5, 1, -2, -2, 0, -2, -2, 4, 1, 3, -2, -2, 2, -2, -2, 5, 2, -2, -2, 0, -2, -2, 0, 4, 3, 2, -2, -2, 6, -2, -2, 3, 2, -2, -2, 3, -2, -2, 4, 2, 2, -2, -2, 2, -2, -2, 2, 3, -2, -2, 4, -2, -2
};

/* Tree node thresholds (scaled feature space) */
static const float ids_threshold[IDS_N_NODES] = {
    1.863929f, 0.502342f, -0.585540f, -0.144838f, -0.211750f, -2.000000f, -2.000000f, 1.276257f, -2.000000f, -2.000000f, -1.084582f, -0.894427f, -2.000000f, -2.000000f, 0.000000f, -2.000000f, -2.000000f, 1.253301f, -0.894427f, 0.589072f, -2.000000f, -2.000000f, 1.381381f, -2.000000f, -2.000000f, 0.948765f, 0.612321f, -2.000000f, -2.000000f, -0.292770f, -2.000000f, -2.000000f, 0.585540f, 0.690343f, 1.532225f, 1.882778f, -2.000000f, -2.000000f, 3.754576f, -2.000000f, -2.000000f, 1.110557f, 2.037344f, -2.000000f, -2.000000f, 1.232737f, -2.000000f, -2.000000f, 1.145300f, 2.778130f, 1.886548f, -2.000000f, -2.000000f, 3.177740f, -2.000000f, -2.000000f, 2.499157f, 1.216032f, -2.000000f, -2.000000f, 1.253301f, -2.000000f, -2.000000f
};

/* Left child node indices (-1 means no child) */
static const int16_t ids_left[IDS_N_NODES] = {
    1, 2, 3, 4, 5, -1, -1, 8, -1, -1, 11, 12, -1, -1, 15, -1, -1, 18, 19, 20, -1, -1, 23, -1, -1, 26, 27, -1, -1, 30, -1, -1, 33, 34, 35, 36, -1, -1, 39, -1, -1, 42, 43, -1, -1, 46, -1, -1, 49, 50, 51, -1, -1, 54, -1, -1, 57, 58, -1, -1, 61, -1, -1
};

/* Right child node indices (-1 means no child) */
static const int16_t ids_right[IDS_N_NODES] = {
    32, 17, 10, 7, 6, -1, -1, 9, -1, -1, 14, 13, -1, -1, 16, -1, -1, 25, 22, 21, -1, -1, 24, -1, -1, 29, 28, -1, -1, 31, -1, -1, 48, 41, 38, 37, -1, -1, 40, -1, -1, 45, 44, -1, -1, 47, -1, -1, 56, 53, 52, -1, -1, 55, -1, -1, 60, 59, -1, -1, 62, -1, -1
};

/* Leaf node class labels (-1 means not a leaf) */
static const int8_t ids_leaf_class[IDS_N_NODES] = {
    -1, -1, -1, -1, -1, 1, 0, -1, 1, 1, -1, -1, 0, 0, -1, 1, 0, -1, -1, -1, 0, 1, -1, 0, 1, -1, -1, 0, 0, -1, 0, 1, -1, -1, -1, -1, 1, 1, -1, 1, 0, -1, -1, 0, 1, -1, 1, 1, -1, -1, -1, 0, 1, -1, 0, 1, -1, -1, 0, 0, -1, 0, 1
};

/*
 * Decision tree inference
 * @param x  Scaled feature vector (15 floats)
 * @return   0 = normal, 1 = attack
 */
static inline int ids_predict(const float *x) {
    int16_t node = 0;
    while (ids_feature[node] >= 0) {
        int16_t f = ids_feature[node];
        if (x[f] <= ids_threshold[node])
            node = ids_left[node];
        else
            node = ids_right[node];
    }
    return ids_leaf_class[node];
}
