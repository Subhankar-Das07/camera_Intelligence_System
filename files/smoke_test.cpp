// Minimal smoke test: not a unit test suite, just proves the two files
// link together and the state machine reaches CONFIRMED on a synthetic
// "person collapses and stays down" sequence, and does NOT fire on a
// synthetic "person sits down slowly" sequence.
#include <cstdio>
#include "fall_state_machine.hpp"
#include "frame_queue.hpp"

using namespace falldet;

// hip_x/hip_y and shoulder_x/shoulder_y are given separately so the test can
// simulate genuine torso *rotation* (shoulder swinging sideways relative to
// the hip), not just both points sliding straight down together — the
// latter never changes the torso angle and would make angular velocity
// (correctly) never fire.
static Keypoints make_kp(float hip_x, float hip_y, float shoulder_x, float shoulder_y, float ankle_y) {
    Keypoints kp;
    for (auto& c : kp.conf) c = 0.0f;
    kp.pts[11] = {hip_x - 5, hip_y}; kp.pts[12] = {hip_x + 5, hip_y}; kp.conf[11] = kp.conf[12] = 0.9f;
    kp.pts[5]  = {shoulder_x - 5, shoulder_y}; kp.pts[6] = {shoulder_x + 5, shoulder_y}; kp.conf[5] = kp.conf[6] = 0.9f;
    kp.pts[15] = {hip_x - 5, ankle_y}; kp.pts[16] = {hip_x + 5, ankle_y}; kp.conf[15] = kp.conf[16] = 0.9f;
    return kp;
}

int main() {
    // --- Queue sanity check ---
    BoundedDropOldestQueue<int> q(2);
    q.push(1); q.push(2); q.push(3);  // 1 should be dropped
    auto a = q.pop(10), b = q.pop(10);
    printf("[queue] popped=%d,%d dropped_count=%zu\n", *a, *b, q.dropped_count());

    // --- Real fall: fast collapse, then stays flat & still ---
    {
        FallConfig cfg;
        FallStateMachine sm(cfg);
        double fps = 30.0;
        bool fired = false;
        // upright baseline: hip at (100,200), shoulder directly above at (100,100)
        // -> dx=0, dy=100 -> torso_angle ~= 0 deg (upright)
        for (int f = 0; f < 10; ++f) sm.update(1, f, fps, make_kp(100, 200, 100, 100, 400), 300, 0.01f);
        // rapid collapse over 6 frames (0.2s): hip drops toward the floor AND
        // the shoulder swings sideways so torso rotates from vertical (~0deg)
        // to flat (~90deg) -- this is what a real fall's angular velocity
        // signal looks like, plus a motion-energy spike at the moment of impact.
        for (int f = 10; f < 16; ++f) {
            const float t = static_cast<float>(f - 9) / 6.f;  // 0..1
            const float hip_y = 200 + t * 190;                // 200 -> 390
            const float sh_x  = 100 + t * 130;                // shoulder swings sideways
            const float sh_y  = 100 + t * 285;                // 100 -> 385 (near hip level)
            auto ev = sm.update(1, f, fps, make_kp(100, hip_y, sh_x, sh_y, 400), 300, 0.5f);
            if (ev) fired = true;
        }
        // now flat on the floor and still for the confirm+stillness window
        for (int f = 16; f < 60; ++f) {
            auto ev = sm.update(1, f, fps, make_kp(100, 390, 230, 385, 400), 300, 0.001f);
            if (ev) { fired = true; printf("[real fall] CONFIRMED at t=%.2fs\n", ev->timestamp_sec); }
        }
        printf("[real fall] fired=%s\n", fired ? "YES (expected)" : "NO (unexpected)");
    }

    // --- Sitting down slowly: should NOT fire ---
    {
        FallConfig cfg;
        FallStateMachine sm(cfg);
        double fps = 30.0;
        bool fired = false;
        for (int f = 0; f < 10; ++f) sm.update(2, f, fps, make_kp(100, 200, 100, 100, 400), 300, 0.01f);
        // slow controlled descent over 3 seconds (90 frames): hip and
        // shoulder move down together (torso stays upright, low angular
        // velocity) and motion stays low the whole time — a real sit-down.
        for (int f = 10; f < 100; ++f) {
            const float t = static_cast<float>(f - 9) / 90.f;
            const float hip_y = 200 + t * 190;
            const float sh_y  = 100 + t * 190;
            auto ev = sm.update(2, f, fps, make_kp(100, hip_y, 100, sh_y, 400), 300, 0.015f);
            if (ev) fired = true;
        }
        printf("[slow sit] fired=%s\n", fired ? "YES (unexpected)" : "NO (expected)");
    }

    return 0;
}
