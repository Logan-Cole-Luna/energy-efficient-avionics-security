/*
 * CAN IDS Firmware
 * 
 * Intrusion Detection System for CubeSat CAN bus
 * Based on Zubax Babel hardware (STM32F373)
 *
 * Original SLCAN code Copyright (C) 2015 Zubax Robotics <info@zubax.com>
 * IDS modifications Copyright (C) 2024
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <ch.hpp>
#include <hal.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <zubax_chibios/os.hpp>
#include <zubax_chibios/util/base64.hpp>
#include <chprintf.h>

#include "board/board.hpp"
#include "usb_cdc.hpp"
#include "ids_protocol.hpp"
#include "ids_inference.hpp"


namespace app
{
namespace
{
/**
 * This is the Brickproof Bootloader's app descriptor.
 * Details: https://github.com/PX4/Firmware/tree/nuttx_next/src/drivers/bootloaders/src/uavcan
 */
static const volatile struct __attribute__((packed))
{
    std::uint8_t signature[8]   = {'A','P','D','e','s','c','0','0'};
    std::uint64_t image_crc     = 0;
    std::uint32_t image_size    = 0;
    std::uint32_t vcs_commit    = GIT_HASH;
    std::uint8_t major_version  = FW_VERSION_MAJOR;
    std::uint8_t minor_version  = FW_VERSION_MINOR;
    std::uint8_t reserved[6]    = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
} _app_descriptor __attribute__((section(".app_descriptor")));


constexpr unsigned WatchdogTimeoutMSec = 1500;

/* UART baudrate config (kept for serial interface) */
os::config::Param<unsigned> cfg_baudrate("uart.baudrate", SERIAL_DEFAULT_BITRATE, 2400, 3000000);


auto init()
{
    /*
     * Basic initialization
     */
    auto watchdog = board::init(WatchdogTimeoutMSec, cfg_baudrate);

    /*
     * USB initialization
     */
    const auto uid = board::readUniqueID();

    usb_cdc::DeviceSerialNumber sn;
    std::fill(sn.begin(), sn.end(), 0);
    std::copy(uid.begin(), uid.end(), sn.begin());

    watchdog.reset();
    usb_cdc::init(sn);                  // Must not exceed watchdog timeout
    watchdog.reset();

    /* Initialize IDS DWT timer */
    ids::initDWT();

    return watchdog;
}

/* Track IDS activity for LED indication */
static volatile bool g_ids_active = false;
static volatile unsigned g_ids_blink_counter = 0;

class BackgroundThread : public chibios_rt::BaseStaticThread<512>
{
    static constexpr unsigned BaseFrameMSec = 25;

    static std::pair<unsigned, unsigned> getStatusOnOffDurationMSec()
    {
        /* IDS ready - slow blink */
        auto& handler = ids_protocol::getHandler();
        if (handler.getState() == ids_protocol::State::ReceiveSamples)
        {
            /* Processing samples - fast blink */
            return {100, 100};
        }
        /* Idle - slow heartbeat */
        return {50, 950};
    }

    static void updateLED()
    {
        /* Traffic LED - blink during inference */
        if (g_ids_blink_counter > 0)
        {
            board::setTrafficLED(true);
            g_ids_blink_counter--;
        }
        else
        {
            board::setTrafficLED(false);
        }

        /* Status LED - heartbeat / activity indicator */
        static bool status_on = false;
        static unsigned status_remaining = 0;

        if (status_remaining > 0)
        {
            status_remaining -= 1;
        }

        if (status_remaining <= 0)
        {
            const auto onoff = getStatusOnOffDurationMSec();
            if (onoff.first == 0 || status_on)
            {
                board::setStatusLED(false);
                status_on = false;
                status_remaining = onoff.second / BaseFrameMSec;
            }
            else
            {
                board::setStatusLED(true);
                status_on = true;
                status_remaining = onoff.first / BaseFrameMSec;
            }
        }
    }

    static void reloadConfigs()
    {
        /* Disable CAN power/terminator - not used in IDS mode */
        board::enableCANPower(false);
        board::enableCANTerminator(false);

        board::reconfigureUART(cfg_baudrate.get());
    }

    void main() override
    {
        reloadConfigs();

        ::systime_t next_step_at = chVTGetSystemTime();
        unsigned cfg_modcnt = 0;


        while (true)
        {
            updateLED();

            const unsigned new_cfg_modcnt = os::config::getModificationCounter();
            if (new_cfg_modcnt != cfg_modcnt)
            {
                cfg_modcnt = new_cfg_modcnt;
                reloadConfigs();

                DEBUG_LOG("Saving config... ");
                const int res = configSave();
                DEBUG_LOG("Config save result: %d\n", res);
                (void) res;
            }

            if (os::isRebootRequested())
            {
                ::usleep(10000);
                NVIC_SystemReset();
            }

            next_step_at += MS2ST(BaseFrameMSec);
            os::sleepUntilChTime(next_step_at);
        }
    }
} background_thread_;

}  // anonymous namespace
}  // namespace app


int main()
{
    /*
     * Initializing
     */
    auto watchdog = app::init();

    chibios_rt::BaseThread::setPriority(NORMALPRIO);

    app::background_thread_.start(LOWPRIO);

    /* Print startup banner */
    std::puts("\r\n[IDS] CAN Intrusion Detection System Ready\r\n");
    std::puts("[IDS] Protocol: 'S' + uint16 count to start benchmark\r\n");
    std::puts("[IDS] Commands: bootloader, reboot, ids_info\r\n");

    /*
     * Running the serial port processing loop
     */
    static constexpr unsigned ReadTimeoutMSec = 5;

    const auto usb_port = usb_cdc::getSerialUSBDriver();
    const auto uart_port = &STDOUT_SD;

    auto& ids_handler = ids_protocol::getHandler();

    while (true)
    {
        watchdog.reset();

        ::BaseChannel* const stdio_stream = os::getStdIOStream();

        static std::uint8_t buf[128];

        /* Read as much as possible without blocking to maximize throughput */
        std::size_t nread = chnReadTimeout(stdio_stream, buf, sizeof(buf), TIME_IMMEDIATE);
        if (nread == 0)
        {
            nread = chnReadTimeout(stdio_stream, buf, 1, MS2ST(ReadTimeoutMSec));
        }

        if (nread > 0)
        {
            /* Process received data through IDS protocol handler */
            ids_handler.processData(buf, static_cast<unsigned>(nread));

            /* Blink traffic LED during activity */
            app::g_ids_blink_counter = 2;
        }
        else
        {
            /* Switching interfaces if necessary */
            const bool using_usb = reinterpret_cast<::BaseChannel*>(stdio_stream) ==
                                   reinterpret_cast<::BaseChannel*>(usb_port);
            const bool usb_connected = usb_cdc::getState() == usb_cdc::State::Connected;
            if (using_usb != usb_connected)
            {
                DEBUG_LOG("Switching to %s\n", usb_connected ? "USB" : "UART");
                os::setStdIOStream(usb_connected ?
                                   reinterpret_cast<::BaseChannel*>(usb_port) :
                                   reinterpret_cast<::BaseChannel*>(uart_port));
                ids_handler.reset();
            }
        }
    }
}


#define MATCH_GCC_VERSION(major, minor, patch)  \
    ((__GNUC__ == (major)) && (__GNUC_MINOR__ == (minor)) && (__GNUC_PATCHLEVEL__ == (patch)))

#if !(MATCH_GCC_VERSION(10, 3, 1))
# error "This firmware requires GCC 10.3.1"
#endif
