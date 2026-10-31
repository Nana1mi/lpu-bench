#ifndef LPU_BENCH_TIMER_H
#define LPU_BENCH_TIMER_H

#include "maps_runtime_api.h"
#include <chrono>
#include <cstdint>
#include <iostream>

#if defined(__aurora_s__) || defined(__confidence__)
#include "tick.h"
#endif

// Struct holding multi-dimensional timing measurements
struct TimingResult {
    float sim_ms;               // Simulation milliseconds from mapsEventElapsedTimeEx
    uint64_t sim_cycles;        // Simulation cycles from mapsEventElapsedTimeEx
    uint64_t device_cycles;     // Device SPU hardware cycles from tick() register
    double wall_clock_ms;       // Host physical wall-clock milliseconds
};

#ifdef __cplusplus
class HostEventTimer {
private:
    mapsEvent_t start_event_;
    mapsEvent_t stop_event_;
    mapsStream_t stream_;
    std::chrono::high_resolution_clock::time_point wall_start_;
    std::chrono::high_resolution_clock::time_point wall_stop_;

public:
    HostEventTimer(mapsStream_t stream = 0) : stream_(stream) {
        mapsEventCreate(&start_event_);
        mapsEventCreate(&stop_event_);
    }

    ~HostEventTimer() {
        mapsEventDestroy(start_event_);
        mapsEventDestroy(stop_event_);
    }

    void start() {
        wall_start_ = std::chrono::high_resolution_clock::now();
        mapsEventRecord(start_event_, stream_);
    }

    void stop() {
        mapsEventRecord(stop_event_, stream_);
        mapsEventSynchronize(stop_event_);
        wall_stop_ = std::chrono::high_resolution_clock::now();
    }

    TimingResult get_result(uint64_t dev_cycles = 0) {
        TimingResult res;
        res.sim_ms = 0.0f;
        res.sim_cycles = 0;
        res.device_cycles = dev_cycles;

        mapsEventElapsedTimeEx(&res.sim_ms, &res.sim_cycles, start_event_, stop_event_);
        
        std::chrono::duration<double, std::milli> wall_dur = wall_stop_ - wall_start_;
        res.wall_clock_ms = wall_dur.count();
        return res;
    }
};
#endif

#endif // LPU_BENCH_TIMER_H
