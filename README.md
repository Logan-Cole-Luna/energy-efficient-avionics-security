# Energy-Efficient Intrusion Detection for Satellite CAN Networks

A production-ready, lightweight intrusion detection system (IDS) for satellite Controller Area Network (CAN) bus traffic, engineered for resource-constrained microcontrollers. The system occupies **541 bytes** of flash memory, executes in **4.2 microseconds**, and imposes a power increment of at most **48–53 microWatts**—validated on hardware from the KISPE Satellite Learning Laboratory (SATLL).

This work addresses a critical gap in satellite cybersecurity: protecting energy-grid-dependent spacecraft (GPS timing, renewable energy telemetry, grid monitoring) while respecting the extreme power and memory budgets of CubeSat-class platforms.

We present the following:

A **domain-specific anomaly detection system** for satellite bus networks:

- **14 CAN features** capturing payload structure, timing dynamics, and bus topology
- **TinyDecisionTree** (depth 5, 39 nodes) as the sole model meeting hardware constraints
- **Zero-heap C implementation** for deterministic, bounded inference on bare-metal STM32
- **Power-aware evaluation** using live SATLL EPS telemetry (not simulated estimates)

## Core Results

### In-Domain Performance (Native Satellite CAN)
Training set: 25,861 frames | Test set: 5,166 frames | Attack classes: DoS, Fuzzy, Spoofing, Replay

| Metric | Value |
|--------|-------|
| **Recall** | **99.86%** (4 missed attacks) |
| **Precision** | 96.38% |
| **F1-Score** | 98.09% |
| **ROC-AUC** | **0.9912** |

### Hardware Validation (STM32F373C8T @ 72 MHz)

| Metric | Value |
|--------|-------|
| **Flash (tree + scaler)** | 541 bytes |
| **RAM (feature buffer)** | 60 bytes (stack only) |
| **Inference latency (mean)** | 4.19–4.21 µs |
| **CPU duty @ 100 Hz** | 0.042% |
| **Inference increment** | **≤48–53 µW** |

### Cross-Domain Generalization (UNSW-NB15 & NSL-KDD)
Network benchmark datasets encoded as synthetic CAN streams, then evaluated on STM32:

| Dataset | Recall | Precision | F1 | FPR |
|---------|--------|-----------|----|----|
| UNSW-NB15 (82K test) | 93.64% | 55.85% | 69.97% | 90.68% |
| NSL-KDD (48K test) | 52.88% | 86.65% | 65.68% | 10.76% |

**Interpretation:** The satellite-trained model generalizes reasonably to alien network traffic, but authentic mission data is essential for production training.

## Repository Structure

```
├── EmbeddedBabel/                  # Zubax Babel USB-CAN adapter firmware (STM32F373)
│   ├── firmware/                   # Main application firmware
│   │   ├── src/                    # Application source code
│   │   │   ├── ids_inference.*     # IDS inference engine
│   │   │   ├── ids_model.h         # Exported decision tree model
│   │   │   ├── ids_scaler.h        # Feature normalization constants
│   │   │   └── main.cpp            # Application entry point
│   │   ├── zubax_chibios/          # ChibiOS RTOS submodule
│   │   └── Makefile                # Build configuration
│   ├── bootloader/                 # USB/UART bootloader
│   │   ├── src/                    # Bootloader source
│   │   └── zubax_chibios/          # ChibiOS RTOS submodule
│   ├── docs/                       # Hardware documentation & datasheets
│   ├── hardware/                   # Enclosure STL files
│   └── tools/                      # DrWatson production testing tools
│
├── scripts/                        # Python ML/IDS pipeline scripts
│   ├── src/                        # Shared utilities (imported by pipeline scripts)
│   │   ├── features.py             # 14-feature CAN extractor + FEATURE_NAMES
│   │   ├── encode_to_can.py        # UNSW/NSL tabular → CAN frame stream encoder
│   │   └── export_firmware.py      # QuantizedDecisionTree + generate_c_header
│   ├── train_cross_domain.py       # Main pipeline: encode → extract → train → export
│   ├── compare_models.py           # 5-model benchmark on UNSW/NSL (model selection)
│   ├── generate_plots.py           # All paper figures
│   ├── run_benchmark.py            # STM32 UART hardware benchmark driver
│   └── export_scaler.py            # Export scaler constants as C header
│
├── models/
│   ├── lightweight_ids_models.py   # Model definitions
│   └── trained_models/
│       ├── satellite/              # Satellite CAN IDS (native data)
│       │   ├── tree.joblib
│       │   ├── scaler.joblib
│       │   ├── features.json
│       │   ├── float32.h
│       │   ├── int16.h
│       │   └── scaler.h
│       ├── unsw/                   # UNSW-NB15 network model
│       │   ├── tree.joblib, scaler.joblib, features.json
│       │   ├── float32.h, int16.h, int8.h
│       ├── nsl/                    # NSL-KDD network model
│       │   ├── tree.joblib, scaler.joblib, features.json
│       │   ├── float32.h, int16.h
│       ├── unsw_can/               # UNSW encoded as CAN frames
│       │   ├── tree.joblib, scaler.joblib, features.json
│       │   ├── float32.h, int16.h
│       └── nsl_can/                # NSL encoded as CAN frames
│           ├── tree.joblib, scaler.joblib, features.json
│           ├── float32.h, int16.h, scaler.h
│
├── firmware/                       # Exported STM32 model headers (copied from models/)
│
├── datasets/
│   ├── UNSW_NB15_training-set.parquet
│   ├── UNSW_NB15_testing-set.parquet
│   ├── NSL-KDD/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── CAN_FROM_BENCHMARK/         # Encoded CAN frame CSVs (auto-generated)
│
└── results/
    ├── model_comparison.json
    ├── model_selection_unsw.json
    ├── model_selection_nsl.json
    ├── unsw_can_results.json
    ├── nsl_can_results.json
    ├── quantization_results.json
    └── plots/
```

## How It Works

### 1. CAN Feature Extraction

Maintain a 50-frame sliding window per CAN ID. For each frame, compute 14 features:

**Payload structure (per-frame):**
- `can_id_norm` — Normalized 11-bit CAN ID
- `dlc` — Data length code (0–8 bytes)
- `data_mean`, `data_std` — Payload byte statistics
- `data_entropy` — Shannon entropy (bits)
- `data_range` — max − min of bytes
- `hamming_dist` — Bit differences from previous frame with same ID
- `payload_delta` — L1 distance from previous payload
- `dlc_anomaly`, `id_is_known` — Anomaly flags

**Temporal dynamics (windowed):**
- `inter_arrival_mean` — Mean Δt between same-ID frames
- `id_freq` — This ID's message rate (msgs/s)

**Bus topology (windowed):**
- `bus_load` — Total bus message rate (msgs/s)
- `unique_ids` — Distinct CAN IDs in window

### 2. Model Selection & Training

Five candidates (TinyDecisionTree, TinyXGBoost, MicroXGBoost, LightRandomForest, CompactExtraTrees) evaluated on UNSW-NB15 and NSL-KDD. **TinyDecisionTree selected** as the only model meeting all three constraints:
- ≤ 8 KB serialized flash
- ≤ 100 µs inference latency
- ≥ 65% recall on both datasets

### 3. Firmware Export & Integration

**Exported as static C arrays** (no heap, no FPU calls):
- `ids_threshold[]`, `ids_feature[]`, `ids_left[]`, `ids_right[]` — decision structure
- `ids_feat_min[]`, `ids_feat_scale[]` (14 × float32) — normalization constants
- Total: 541 bytes in flash; 60 bytes stack for feature buffer

**Root-to-leaf traversal:** Deterministic 302-cycle fixed path.

### 4. Power Accounting

**Two-level decomposition:**

| Component | Draw |
|-----------|------|
| MCU base (STM32F373C8T @ 72 MHz) | 115–125 mW |
| Board peripherals (USB-FS, LDO) | ≈49 mW |
| **Inference algorithm (100 Hz, 4.2 µs)** | **≤48–53 µW** |

The 48–53 µW figure is the **marginal cost of tree traversal alone**, derived from live SATLL EPS telemetry. It represents **<0.004% of ADCS subsystem overhead** during satellite experiments.

## Getting Started

### Prerequisites

```bash
cd intrusion_detection
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib xgboost matplotlib seaborn
```

### 1. Download and Prepare Datasets

```bash
# Place parquet files in datasets/
# UNSW_NB15_training-set.parquet, UNSW_NB15_testing-set.parquet

# NSL-KDD text files in datasets/NSL-KDD/
# KDDTrain+.txt, KDDTest+.txt
```

### 2. Run Model Selection Benchmark (optional)

Trains all 5 candidate models on UNSW and NSL to confirm TinyDecisionTree selection:

```bash
python scripts/compare_models.py
```

Produces:
- `results/model_selection_unsw.json`
- `results/model_selection_nsl.json`
- `results/model_comparison.json`
- `results/plots/all_models_comparison.png`

### 3. Train Cross-Domain CAN IDS

Encodes UNSW or NSL as CAN frame streams, extracts 14 features, trains TinyDecisionTree, and exports STM32 C headers to both `models/trained_models/{dataset}_can/` and `firmware/`:

```bash
python scripts/train_cross_domain.py --dataset unsw
python scripts/train_cross_domain.py --dataset nsl
```

Produces per dataset:
- `models/trained_models/{dataset}_can/tree.joblib`
- `models/trained_models/{dataset}_can/scaler.joblib`
- `models/trained_models/{dataset}_can/features.json`
- `models/trained_models/{dataset}_can/float32.h` (also copied to `firmware/`)
- `models/trained_models/{dataset}_can/int16.h` (also copied to `firmware/`)
- `results/{dataset}_can_results.json`

### 4. Evaluate on Hardware (SATLL + STM32)

Flash firmware with embedded model header to STM32F373C8T evaluation board, powered from SATLL payload rail:

```bash
python scripts/run_benchmark.py \
  --port /dev/tty.usbmodem12345 \
  --features datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
  --labels datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv
```

### 5. Generate Paper Figures

```bash
python scripts/generate_plots.py
```

Outputs all figures to `results/plots/`:
- `model_comparison.png`, `tree_structure_nsl.png`, `tree_structure_can.png`
- `feature_importance.png`, `roc_curves.png`, `radar_summary.png`
- `benchmark_can_comparison.png`, `quantization_metrics.png`, etc.

## Key Features

**Minimal resource footprint** — 541 bytes flash, 60 bytes RAM  
**Deterministic latency** — 4.2 µs fixed-path tree traversal  
**Negligible power overhead** — ≤53 µW inference increment  
**High in-domain accuracy** — 99.86% recall on satellite CAN  
**Production-grade firmware** — No heap, no malloc, stack-only  
**Realistic power accounting** — Measured from actual spacecraft EPS telemetry  
**Cross-domain validation** — Tested on UNSW-NB15 and NSL-KDD  

**Tested platform:** STM32F373C8T at 72 MHz (as used in SATLL OBDH-class hardware)

## Performance Characteristics

### Latency Profile

| Operation | Time |
|-----------|------|
| Feature computation (14 features) | ~2 µs |
| Z-score normalization | ~1 µs |
| Tree traversal (5 levels) | ~1.2 µs |
| **Total inference** | **~4.2 µs** |

### Memory Profile

| Resource | Size | Notes |
|----------|------|-------|
| Tree arrays (429 B) | 429 bytes | Thresholds, split indices, leaves |
| Scaler constants (112 B) | 112 bytes | 14 × float32 mean/scale |
| Feature buffer | 60 bytes | Stack-allocated, reused per sample |
| **Total flash** | **541 bytes** | |
| **Total RAM (dynamic)** | **0 bytes** | No heap allocation |

### Power Profile

At 100 Hz polling rate with STM32F373C8T at 72 MHz:

| Metric | Value |
|--------|-------|
| MCU base current (active-run) | 35–38 mA @ 3.3 V |
| MCU base power | 115–125 mW |
| Inference margin | ≤48–53 µW per inference |
| Total evaluation board | 164–176 mW (external MCU + bridge) |
| Native OBDH integration projection | ~40 µW residual |

**Power data source:** Live SATLL EPS telemetry (1 Hz CDH logging); not simulated or estimated.

## Reproducing Results

### Step 1: Model Selection

```bash
python scripts/compare_models.py
```

### Step 2: Train CAN IDS

```bash
python scripts/train_cross_domain.py --dataset unsw
python scripts/train_cross_domain.py --dataset nsl
```

Expected output (NSL):
```
Recall   : 0.5288
FPR      : 0.1076
ROC-AUC  : 0.7951
```

Expected output (UNSW):
```
Recall   : 0.9364
FPR      : 0.9068
ROC-AUC  : 0.9301
```

### Step 3: Build & Flash STM32 Firmware (Zubax Babel)

The IDS runs on [Zubax Babel](https://zubax.com/products/babel) hardware—a USB-CAN adapter based on STM32F373.

**Prerequisites:**
- ARM GCC toolchain (see `EmbeddedBabel/firmware/src/main.cpp` for required version)
- Initialize submodules: `git submodule update --init --recursive`

**Building the firmware:**

```bash
cd EmbeddedBabel/firmware
make -j8 RELEASE=1   # Omit RELEASE=1 for debug build
```

Build outputs in `EmbeddedBabel/firmware/build/`:
- `com.zubax.*.application.bin` — Application binary for bootloader upload
- `com.zubax.*.compound.bin` — Combined bootloader + application for empty MCU
- `compound.elf` — Debug symbols for SWD debugging

**Flashing via SWD/JTAG (recommended):**

Use a [Zubax Dronecode Probe](https://kb.zubax.com/x/iIAh) or any SWD debugger:

```bash
cd EmbeddedBabel/firmware
./zubax_chibios/tools/blackmagic_flash.sh
```

Or with `st-flash`:

```bash
st-flash write build/com.zubax.babel.compound.bin 0x08000000
```

**Flashing via USB/UART Bootloader:**

1. Connect to the device CLI (USB virtual serial port or UART at 115200-8N1)
2. Execute `bootloader` command to enter bootloader mode
3. Execute `download` to start XMODEM receiver
4. Transmit the application binary:

```bash
sz -vv --xmodem --1k EmbeddedBabel/firmware/build/com.zubax.babel.application.bin > /dev/ttyACM0 < /dev/ttyACM0
```

Or use the automated script:

```bash
cd EmbeddedBabel/firmware
./zubax_chibios/tools/flash_via_serial_bootloader.sh
```

**Verify installation:**

Connect via serial and run `zubax_id` — if in application mode, IDS is ready.

### Step 4: Hardware Validation

```bash
python scripts/run_benchmark.py \
  --port /dev/tty.usbmodem12345 \
  --features datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
  --labels datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
  --output results/hardware_validation.json
```

Expected matching:
- **Hardware latency:** 4.19–4.21 µs 
- **Inference increment:** 48–53 µW 
- **In-domain recall:** 99.86% 
- **CPU duty @ 100 Hz:** 0.042% 

## Future Work

1. **Direct OBDH integration** — Deploy natively on STM32H573 OBDH firmware (currently prevented by closed-source SATLL software; future open-firmware platforms targeted)
2. **Dynamic thresholding** — Adapt detection boundary at runtime based on operational context (eclipse, maneuver, nominal)
3. **Lightweight ensemble** — Combine multiple shallow trees within the same 541-byte flash budget for adversarial robustness
4. **Operational traffic collection** — Expand training dataset to include eclipse transitions, orbit corrections, and payload activations
5. **Hardware acceleration** — Co-design with DSP accelerators for multi-satellite constellation monitoring

## License

[Specify: MIT, Apache 2.0, or other as appropriate]
