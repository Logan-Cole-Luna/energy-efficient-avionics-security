/*
 * IDS Protocol Handler
 * 
 * Binary protocol for IDS benchmark over USB CDC:
 *   - Start: 'S' + uint16 sample count
 *   - Per sample: 14×float32 (56 bytes) + uint8 label
 *   - Response: 16-byte SampleResult per sample, 68-byte BenchSummary at end
 *   - Command: "bootloader\r" to enter bootloader
 *
 * Copyright (C) 2024
 */

#pragma once

#include <cstdint>

namespace ids_protocol
{

/**
 * Protocol handler state machine
 */
enum class State {
    Idle,           /* Waiting for 'S' start command or text command */
    WaitSampleCount,/* Waiting for uint16 sample count */
    ReceiveSamples, /* Receiving features + labels, sending results */
    Done            /* Batch complete, summary sent */
};

/**
 * Protocol handler for IDS benchmark communication
 */
class ProtocolHandler {
public:
    ProtocolHandler();
    
    /**
     * Process incoming data from USB/UART
     * @param data   Pointer to received bytes
     * @param len    Number of bytes received
     */
    void processData(const std::uint8_t* data, unsigned len);
    
    /**
     * Reset protocol state (e.g., on interface switch)
     */
    void reset();
    
    /**
     * Get current protocol state
     */
    State getState() const { return state_; }

private:
    State state_;
    
    /* Sample count for current batch */
    std::uint16_t n_samples_;
    std::uint16_t samples_received_;
    
    /* Receive buffer for multi-byte data */
    static constexpr unsigned BUF_SIZE = 256;
    std::uint8_t buf_[BUF_SIZE];
    unsigned buf_pos_;
    
    /* Benchmark statistics */
    std::uint32_t tp_, tn_, fp_, fn_;
    std::uint32_t cycles_min_, cycles_max_;
    std::uint64_t cycles_sum_;
    std::uint32_t stack_max_;
    
    /* Text command buffer for "bootloader" */
    char cmd_buf_[16];
    unsigned cmd_pos_;
    
    /* Process a complete text command line */
    void processCommand(const char* cmd);
    
    /* Process start command with sample count */
    void processStartCommand();
    
    /* Process one complete sample (features + label) */
    void processSample();
    
    /* Finalize batch and send summary */
    void finalizeBatch();
};

/**
 * Global protocol handler instance
 */
ProtocolHandler& getHandler();

}  // namespace ids_protocol
