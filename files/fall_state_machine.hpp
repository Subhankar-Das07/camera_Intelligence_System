// fall_state_machine.hpp
//
// C++ port of the Python fall-detection state machine, meant to be called
// from a DeepStream tracker src-pad probe (or any native pipeline that
// gives you per-track COCO-17 keypoints + a bbox height + a motion-energy
// value each frame). Logic is kept in lockstep with
// fall_detection_pipeline_v2.py so tuning one mirrors the other.
//
// Same four-signal AND-gate as the Python version:
//   hip drop velocity (Kalman) + torso angular velocity + motion-energy
//   spike + track age/confidence all have to agree before a track is even
//   considered "falling", and posture-down + stillness both have to hold
//   before it's "confirmed".

#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace falldet {

struct Point2f {
    float x = 0.f;
    float y = 0.f;
};

// COCO-17 layout; conf < 0 means "not detected this frame".
struct Keypoints {
    Point2f pts[17];
    float conf[17] = {0};
};

enum class TrackState : std::uint8_t { Normal = 0, Falling = 1, Confirmed = 2 };

// Minimal constant-velocity Kalman filter for a single noisy scalar (hip_y).
struct Kalman1D {
    double x = 0.0, v = 0.0;
    double P[2][2] = {{100.0, 0.0}, {0.0, 100.0}};
    double q = 4.0;   // process variance
    double r = 6.0;   // measurement variance
    bool initialized = false;

    void reset(double x0) {
        x = x0;
        v = 0.0;
        P[0][0] = P[1][1] = 100.0;
        P[0][1] = P[1][0] = 0.0;
        initialized = true;
    }

    // Returns {position, velocity}.
    std::pair<double, double> update(double z, double dt);
};

struct FallConfig {
    float velocity_threshold = 0.55f;              // normalized hip-drop vel
    float angular_velocity_threshold_deg_s = 90.f;  // torso-angle change rate
    float motion_spike_ratio = 1.8f;                // spike vs local baseline
    float motion_spike_floor = 0.01f;
    float sit_angle_deg = 30.f;
    float down_torso_angle_deg = 40.f;
    float down_hip_ankle_ratio = 0.45f;
    int   confirm_frames = 8;
    int   min_track_age_frames = 5;
    float min_keypoint_conf = 0.35f;
    bool  require_stillness_confirmation = true;
    int   stillness_frames = 10;
    float stillness_motion_threshold = 0.02f;
    std::size_t angle_history_len = 15;
    std::size_t motion_history_len = 15;
};

struct AlertEvent {
    std::int64_t track_id = -1;
    double timestamp_sec = 0.0;
    std::string reason;
    // Signal snapshot at the moment of confirmation, for audit/debugging —
    // mirrors the `signals` block in the Python pipeline's alert payload.
    float drop_vel_norm = 0.f;
    float angular_vel_deg_s = 0.f;
    float motion_energy = 0.f;
    float torso_angle_deg = 0.f;
};

struct PersonState {
    TrackState state = TrackState::Normal;
    Kalman1D hip_kf;
    std::deque<float> angle_history;
    std::deque<float> motion_history;
    double baseline_h = -1.0;
    int down_frame_count = 0;
    int still_frame_count = 0;
    int track_age = 0;
    std::int64_t last_frame_idx = -1;
};

class FallStateMachine {
public:
    explicit FallStateMachine(FallConfig cfg) : cfg_(std::move(cfg)) {}

    // Call once per frame per tracked person (e.g. from a DeepStream tracker
    // src-pad probe, iterating NvDsObjectMeta for class_id == PERSON).
    // `motion_energy_norm` should be a 0..~1 normalized motion signal for
    // the person's ROI (frame-diff mean or NVOF magnitude both work).
    // Returns an AlertEvent on the exact frame a fall is confirmed.
    std::optional<AlertEvent> update(std::int64_t track_id,
                                      std::int64_t frame_idx,
                                      double fps,
                                      const Keypoints& kp,
                                      float bbox_h,
                                      float motion_energy_norm);

    // Call once per frame to drop tracks DeepStream's tracker has lost.
    void evict_stale(std::int64_t frame_idx, std::int64_t max_age_frames);

    std::size_t active_track_count() const { return states_.size(); }

private:
    FallConfig cfg_;
    std::unordered_map<std::int64_t, PersonState> states_;

    static bool mid_point(const Keypoints& kp, int li, int ri, float thresh,
                           Point2f& out, float& conf_out);
    static float torso_angle_deg(const Point2f& hip, const Point2f& shoulder);
};

}  // namespace falldet
