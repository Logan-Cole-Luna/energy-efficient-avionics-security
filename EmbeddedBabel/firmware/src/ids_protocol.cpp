/*
 * IDS Protocol Handler Implementation
 * 
 * Binary protocol for IDS benchmark over USB CDC
 *
 * Copyright (C) 2024
 */

#include "ids_protocol.hpp"
#include "ids_inference.hpp"

#include <ch.hpp>
#include <zubax_chibios/os.hpp>
#include <cstring>
#include <cstdio>
#include <climits>

/* Bootloader interface */
#include "../../bootloader/src/bootloader_app_interface.hpp"

namespace ids_protocol
{

/* Feature + label size: 14 floats + 1 byte = 57 bytes */
constexpr unsigned SAMPLE_INPUT_SIZE = ids::N_FEATURES * sizeof(float) + 1;

/* Timeout for receiving data */
constexpr unsigned WRITE_TIMEOUT_MS = 50;

static ProtocolHandler s_handler;

ProtocolHandler& getHandler()
{
    return s_handler;
}

ProtocolHandler::ProtocolHandler()
    : state_(State::Idle)
    , n_samples_(0)
    , samples_received_(0)
    , buf_pos_(0)
    , tp_(0), tn_(0), fp_(0), fn_(0)
    , cycles_min_(UINT32_MAX), cycles_max_(0)
    , cycles_sum_(0)
    , stack_max_(0)
    , cmd_pos_(0)
{
    std::memset(buf_, 0, sizeof(buf_));
    std::memset(cmd_buf_, 0, sizeof(cmd_buf_));
}

void ProtocolHandler::reset()
{
    state_ = State::Idle;
    n_samples_ = 0;
    samples_received_ = 0;
    buf_pos_ = 0;
    cmd_pos_ = 0;
    std::memset(buf_, 0, sizeof(buf_));
    std::memset(cmd_buf_, 0, sizeof(cmd_buf_));
}

void ProtocolHandler::processData(const std::uint8_t* data, unsigned len)
{
    for (unsigned i = 0; i < len; i++) {
        std::uint8_t byte = data[i];
        
        switch (state_) {
        case State::Idle:
            if (byte == 'S') {
                /* Start command - wait for sample count */
                state_ = State::WaitSampleCount;
                buf_pos_ = 0;
            } else if (byte >= 32 && byte <= 126) {
                /* Printable ASCII - collect for text command */
                if (cmd_pos_ < sizeof(cmd_buf_) - 1) {
                    cmd_buf_[cmd_pos_++] = static_cast<char>(byte);
                }
            } else if (byte == '\r' || byte == '\n') {
                /* End of text command */
                cmd_buf_[cmd_pos_] = '\0';
                if (cmd_pos_ > 0) {
                    processCommand(cmd_buf_);
                }
                cmd_pos_ = 0;
            } else if (byte == 8 || byte == 127) {
                /* Backspace */
                if (cmd_pos_ > 0) cmd_pos_--;
            }
            break;
            
        case State::WaitSampleCount:
            buf_[buf_pos_++] = byte;
            if (buf_pos_ >= 2) {
                processStartCommand();
            }
            break;
            
        case State::ReceiveSamples:
            buf_[buf_pos_++] = byte;
            if (buf_pos_ >= SAMPLE_INPUT_SIZE) {
                processSample();
                buf_pos_ = 0;
                
                if (samples_received_ >= n_samples_) {
                    finalizeBatch();
                }
            }
            break;
            
        case State::Done:
            /* Reset to idle and reprocess this byte */
            reset();
            /* Fall through to process the byte in Idle state */
            if (byte == 'S') {
                /* Start command - wait for sample count */
                state_ = State::WaitSampleCount;
                buf_pos_ = 0;
            } else if (byte >= 32 && byte <= 126) {
                /* Printable ASCII - collect for text command */
                if (cmd_pos_ < sizeof(cmd_buf_) - 1) {
                    cmd_buf_[cmd_pos_++] = static_cast<char>(byte);
                }
            } else if (byte == '\r' || byte == '\n') {
                /* End of text command */
                cmd_buf_[cmd_pos_] = '\0';
                if (cmd_pos_ > 0) {
                    processCommand(cmd_buf_);
                }
                cmd_pos_ = 0;
            }
            break;
        }
    }
}

void ProtocolHandler::processCommand(const char* cmd)
{
    if (std::strcmp(cmd, "bootloader") == 0) {
        /* Enter bootloader */
        std::puts("bootloader");
        std::puts("Entering bootloader...");
        std::puts("\x03\r\n");  /* End of multi-line response marker */
        
        bootloader_app_interface::AppShared apsh;
        apsh.stay_in_bootloader = true;
        bootloader_app_interface::write(apsh);
        
        os::requestReboot();
    } else if (std::strcmp(cmd, "reboot") == 0) {
        /* Reboot without bootloader */
        std::puts("reboot");
        std::puts("Rebooting...");
        std::puts("\x03\r\n");
        
        os::requestReboot();
    } else if (std::strcmp(cmd, "ids_info") == 0) {
        /* Print IDS info */
        std::puts("ids_info");
        std::printf("IDS Firmware\n");
        std::printf("  Model: TinyDecisionTree (CAN IDS)\n");
        std::printf("  Features: %u\n", ids::N_FEATURES);
        std::printf("  Tree nodes: 39\n");
        std::printf("  Protocol: Binary (S+count → features+label → results)\n");
        std::puts("\x03\r\n");
    } else {
        /* Unknown command - ignore or send error */
        /* Don't print anything to avoid interfering with binary protocol */
    }
}

void ProtocolHandler::processStartCommand()
{
    /* Parse uint16 little-endian sample count */
    n_samples_ = static_cast<std::uint16_t>(buf_[0]) |
                 (static_cast<std::uint16_t>(buf_[1]) << 8);
    
    if (n_samples_ == 0 || n_samples_ > ids::MAX_SAMPLES) {
        /* Invalid sample count - reset to idle */
        std::puts("[IDS] Bad sample count\r\n");
        reset();
        return;
    }
    
    /* Initialize benchmark state */
    samples_received_ = 0;
    tp_ = tn_ = fp_ = fn_ = 0;
    cycles_min_ = UINT32_MAX;
    cycles_max_ = 0;
    cycles_sum_ = 0;
    stack_max_ = 0;
    buf_pos_ = 0;
    
    /* Paint canary before benchmark loop */
    ids::paintCanary();
    
    /* Initialize DWT cycle counter */
    ids::initDWT();
    
    char logbuf[64];
    std::snprintf(logbuf, sizeof(logbuf), 
                  "[IDS] Starting %u-sample benchmark\r\n", n_samples_);
    std::puts(logbuf);
    
    state_ = State::ReceiveSamples;
}

void ProtocolHandler::processSample()
{
    /* Extract features (14 × float32 = 56 bytes) and label (1 byte) */
    float features[ids::N_FEATURES];
    std::memcpy(features, buf_, sizeof(features));
    std::uint8_t label = buf_[sizeof(features)];
    
    /* Run inference with timing */
    ids::SampleResult result;
    ids::runInference(features, label, result);
    
    /* Update statistics */
    if (result.cycles < cycles_min_) cycles_min_ = result.cycles;
    if (result.cycles > cycles_max_) cycles_max_ = result.cycles;
    cycles_sum_ += result.cycles;
    
    if (result.stack_used_bytes > stack_max_) {
        stack_max_ = result.stack_used_bytes;
    }
    
    /* Confusion matrix */
    if (result.prediction == 1 && label == 1) tp_++;
    else if (result.prediction == 0 && label == 0) tn_++;
    else if (result.prediction == 1 && label == 0) fp_++;
    else fn_++;
    
    samples_received_++;
    
    /* Send result (16 bytes) */
    os::MutexLocker mlocker(os::getStdIOMutex());
    chnWriteTimeout(os::getStdIOStream(),
                    reinterpret_cast<const std::uint8_t*>(&result),
                    sizeof(result), MS2ST(WRITE_TIMEOUT_MS));
}

void ProtocolHandler::finalizeBatch()
{
    /* Compute and send summary */
    ids::BenchSummary summary;
    ids::computeSummary(
        n_samples_,
        tp_, tn_, fp_, fn_,
        cycles_min_, cycles_max_,
        cycles_sum_, stack_max_,
        summary
    );
    
    /* Send summary (68 bytes) */
    {
        os::MutexLocker mlocker(os::getStdIOMutex());
        chnWriteTimeout(os::getStdIOStream(),
                        reinterpret_cast<const std::uint8_t*>(&summary),
                        sizeof(summary), MS2ST(WRITE_TIMEOUT_MS));
    }
    
    /* Log completion */
    char logbuf[128];
    std::snprintf(logbuf, sizeof(logbuf),
                  "[IDS] Done. Acc=%.4f Rec=%.4f FPR=%.4f Inf=%.2fus\r\n",
                  summary.accuracy, summary.recall, summary.fpr,
                  summary.inference_us_mean);
    std::puts(logbuf);
    
    state_ = State::Done;
}

}  // namespace ids_protocol
