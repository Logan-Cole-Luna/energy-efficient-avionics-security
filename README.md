# Energy-Efficient Intrusion Detection for Satellite CAN Networks

A production-ready, lightweight intrusion detection system (IDS) for satellite Controller Area Network (CAN) bus traffic, engineered for resource-constrained microcontrollers.

This repository is now NSL-KDD centric for cross-domain benchmarking and CAN conversion. All benchmark training, model selection, exported artifacts, and plots are generated from NSL-KDD only.

## Core Results (NSL-KDD -> CAN)

| Metric | Value |
|--------|-------|
| Recall | 52.88% |
| Precision | 86.65% |
| F1 | 65.68% |
| FPR | 10.76% |
| ROC-AUC | 0.7951 |

## Repository Structure

```
├── scripts/
│   ├── compare_models.py           # 5-model benchmark on NSL-KDD
│   ├── train.py                    # NSL-KDD -> CAN training + export pipeline
│   ├── generate_plots.py           # NSL-only paper/analysis figures
│   ├── run_benchmark.py            # STM32 UART hardware benchmark driver
│   ├── export_scaler.py            # Export scaler constants as C header
│   └── src/
│       ├── encode_to_can.py        # NSL-KDD tabular -> CAN frame encoder
│       ├── features.py             # 14-feature CAN extractor + FEATURE_NAMES
│       └── export_firmware.py      # QuantizedDecisionTree + generate_c_header
│
├── models/
│   └── trained_models/
│       └── nsl_can/
│           ├── tree.joblib
│           ├── scaler.joblib
│           ├── features.json
│           ├── nsl_can_float32.h
│           └── nsl_can_int16.h
│
├── datasets/
│   ├── NSL-KDD/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── CAN_FROM_BENCHMARK/
│       ├── nsl_can_train.csv
│       ├── nsl_can_test.csv
│       ├── nsl_can_train_features.csv
│       ├── nsl_can_test_features.csv
│       └── nsl_can_encoding_meta.json
│
├── firmware/                       # Exported STM32 model headers
└── results/
    ├── model_selection_nsl.json
    ├── model_comparison.json
    ├── nsl_can_results.json
    └── plots/
```

## Getting Started

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Prepare NSL-KDD Dataset

Place files in `datasets/NSL-KDD/`:

- `KDDTrain+.txt`
- `KDDTest+.txt`

### 2. Run Model Selection on NSL-KDD

```bash
python scripts/compare_models.py
```

Outputs:

- `results/model_selection_nsl.json`
- `results/model_comparison.json`
- `results/plots/all_models_comparison.png`

### 3. Train NSL->CAN IDS and Export Firmware Headers

```bash
python scripts/train.py --dataset nsl
```

Outputs:

- `models/trained_models/nsl_can/tree.joblib`
- `models/trained_models/nsl_can/scaler.joblib`
- `models/trained_models/nsl_can/features.json`
- `models/trained_models/nsl_can/nsl_can_float32.h`
- `models/trained_models/nsl_can/nsl_can_int16.h`
- `firmware/nsl_can_float32.h`
- `firmware/nsl_can_int16.h`
- `results/nsl_can_results.json`

### 4. Generate Plots

```bash
python scripts/generate_plots.py
```

Plots are written to `results/plots/`.

### 5. Optional: Hardware Validation

```bash
python scripts/run_benchmark.py \
  --port /dev/tty.usbmodem12345 \
  --features datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
  --labels datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv
```

## Notes

- The benchmark and CAN conversion pipeline are intentionally restricted to NSL-KDD.
- TinyDecisionTree remains the deployment model due to flash/latency constraints.
- Exported headers are generated with deterministic, zero-heap inference in mind.

## License

[Specify: MIT, Apache 2.0, or other as appropriate]
