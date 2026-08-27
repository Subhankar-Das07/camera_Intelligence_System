// frame_queue.hpp
//
// Thread-safe bounded queue with a drop-oldest policy, for producer/consumer
// video pipelines where the producer (camera decode) must NEVER block on a
// slow consumer (inference). A stale frame is worse than a dropped one for
// realtime fall detection, so this deliberately discards old data instead
// of applying backpressure.
//
// Intended use: one instance per camera stream (single producer thread,
// single consumer thread). For a multi-camera deployment, run one
// capture-thread + one queue per stream feeding into a shared batched
// inference stage (see the architecture note at the bottom of this file).

#pragma once

#include <condition_variable>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

template <typename T>
class BoundedDropOldestQueue {
public:
    explicit BoundedDropOldestQueue(std::size_t capacity) : capacity_(capacity) {}

    // Producer side. Never blocks: if full, silently evicts the oldest item.
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_) {
            queue_.pop_front();
            ++dropped_count_;
        }
        queue_.push_back(std::move(item));
        cv_.notify_one();
    }

    // Consumer side. Blocks up to timeout_ms waiting for an item.
    std::optional<T> pop(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool got = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !queue_.empty() || stopping_; });
        if (!got || queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop_front();
        return item;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        cv_.notify_all();
    }

    std::size_t dropped_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_count_;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    std::size_t capacity_;
    std::size_t dropped_count_ = 0;
    bool stopping_ = false;
};

// ---------------------------------------------------------------------------
// Production architecture note
// ---------------------------------------------------------------------------
// For more than a handful of streams, don't run per-stream YOLO/TensorRT
// inference on separate threads — you'll thrash the GPU context switching
// between small batches. Standard pattern:
//
//   [capture thread + BoundedDropOldestQueue] x N cameras
//                     |
//                     v
//         [single batching inference thread]  -- gathers up to 1 frame per
//                                                 camera per tick, runs one
//                                                 batched TensorRT forward
//                                                 pass, fans results back out
//                     |
//                     v
//   [N post-process threads running FallStateMachine::update, one per track set]
//                     |
//                     v
//         [async clip writer thread pool]  -- never blocks the state machine
//
// At real scale (dozens of outdoor cameras), replace the hand-rolled
// capture+inference threads with NVIDIA DeepStream: nvurisrcbin handles
// RTSP reconnect and hardware decode (NVDEC) per stream, nvstreammux batches
// frames from all streams into one TensorRT forward pass automatically, and
// FallStateMachine::update (fall_state_machine.hpp) plugs in as a pad probe
// on the tracker's src pad. That removes the need to hand-write the
// capture/batching threads at all — DeepStream's C++ pipeline already does
// it, and you keep the fall logic identical to what's in this file set.
