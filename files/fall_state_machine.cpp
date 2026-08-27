// fall_state_machine.cpp
#include "fall_state_machine.hpp"

#include <cmath>
#include <numeric>

namespace falldet {

std::pair<double, double> Kalman1D::update(double z, double dt) {
    if (!initialized || dt <= 0.0) {
        reset(z);
        return {x, v};
    }
    // Predict (constant-velocity model): state = F * state, P = F P F^T + Q
    const double F00 = 1.0, F01 = dt, F10 = 0.0, F11 = 1.0;
    const double px = F00 * x + F01 * v;
    const double pv = F10 * x + F11 * v;

    const double q11 = (dt * dt * dt * dt / 4.0) * q;
    const double q12 = (dt * dt * dt / 2.0) * q;
    const double q22 = (dt * dt) * q;

    const double fp00 = F00 * P[0][0] + F01 * P[1][0];
    const double fp01 = F00 * P[0][1] + F01 * P[1][1];
    const double fp10 = F10 * P[0][0] + F11 * P[1][0];
    const double fp11 = F10 * P[0][1] + F11 * P[1][1];

    const double p00 = fp00 * F00 + fp01 * F01 + q11;
    const double p01 = fp00 * F10 + fp01 * F11 + q12;
    const double p10 = fp10 * F00 + fp11 * F01 + q12;
    const double p11 = fp10 * F10 + fp11 * F11 + q22;

    // Update (measure position only)
    const double yres = z - px;
    const double S = p00 + r;
    const double K0 = p00 / S;
    const double K1 = p10 / S;

    x = px + K0 * yres;
    v = pv + K1 * yres;
    P[0][0] = (1.0 - K0) * p00;
    P[0][1] = (1.0 - K0) * p01;
    P[1][0] = p10 - K1 * p00;
    P[1][1] = p11 - K1 * p01;

    return {x, v};
}

bool FallStateMachine::mid_point(const Keypoints& kp, int li, int ri, float thresh,
                                  Point2f& out, float& conf_out) {
    const bool lo = kp.conf[li] > thresh;
    const bool ro = kp.conf[ri] > thresh;
    if (lo && ro) {
        out.x = (kp.pts[li].x + kp.pts[ri].x) / 2.f;
        out.y = (kp.pts[li].y + kp.pts[ri].y) / 2.f;
        conf_out = (kp.conf[li] + kp.conf[ri]) / 2.f;
        return true;
    }
    if (lo) { out = kp.pts[li]; conf_out = kp.conf[li]; return true; }
    if (ro) { out = kp.pts[ri]; conf_out = kp.conf[ri]; return true; }
    return false;
}

float FallStateMachine::torso_angle_deg(const Point2f& hip, const Point2f& shoulder) {
    const float dx = std::fabs(shoulder.x - hip.x);
    const float dy = std::fabs(hip.y - shoulder.y) + 1e-6f;
    return static_cast<float>(std::atan2(dx, dy) * 180.0 / M_PI);
}

std::optional<AlertEvent> FallStateMachine::update(std::int64_t track_id,
                                                     std::int64_t frame_idx,
                                                     double fps,
                                                     const Keypoints& kp,
                                                     float bbox_h,
                                                     float motion_energy_norm) {
    auto& ps = states_[track_id];
    ps.track_age++;
    const double dt = (ps.last_frame_idx >= 0)
                           ? static_cast<double>(frame_idx - ps.last_frame_idx) / fps
                           : 1.0 / fps;
    ps.last_frame_idx = frame_idx;

    Point2f hip{}, shoulder{}, ankle{};
    float hip_conf = 0.f, sh_conf = 0.f, an_conf = 0.f;
    const bool has_hip = mid_point(kp, 11, 12, 0.4f, hip, hip_conf);
    const bool has_sh  = mid_point(kp, 5, 6, 0.4f, shoulder, sh_conf);
    const bool has_an  = mid_point(kp, 15, 16, 0.4f, ankle, an_conf);
    if (!has_hip) return std::nullopt;  // nothing usable this frame

    if (ps.baseline_h < 0) ps.baseline_h = bbox_h;

    const float torso_angle = has_sh ? torso_angle_deg(hip, shoulder) : -1.f;
    const bool is_upright = has_sh && torso_angle < cfg_.sit_angle_deg;
    if (is_upright) ps.baseline_h = 0.85 * ps.baseline_h + 0.15 * bbox_h;

    const auto [hip_y_filt, hip_vel] = ps.hip_kf.update(hip.y, dt);
    (void)hip_y_filt;
    const float drop_vel_norm = (ps.baseline_h > 1.0)
                                     ? static_cast<float>(hip_vel / ps.baseline_h)
                                     : 0.f;

    ps.angle_history.push_back(torso_angle);
    if (ps.angle_history.size() > cfg_.angle_history_len) ps.angle_history.pop_front();
    float angular_vel = 0.f;
    if (ps.angle_history.size() >= 2 && torso_angle >= 0.f) {
        const float span_sec = static_cast<float>(ps.angle_history.size()) / static_cast<float>(fps);
        angular_vel = (ps.angle_history.back() - ps.angle_history.front()) / span_sec;
    }

    ps.motion_history.push_back(motion_energy_norm);
    if (ps.motion_history.size() > cfg_.motion_history_len) ps.motion_history.pop_front();
    const float motion_baseline =
        std::accumulate(ps.motion_history.begin(), ps.motion_history.end(), 0.f) /
        static_cast<float>(ps.motion_history.size());

    bool just_confirmed = false;

    switch (ps.state) {
        case TrackState::Normal: {
            const bool velocity_ok = drop_vel_norm > cfg_.velocity_threshold;
            const bool angular_ok = std::fabs(angular_vel) > cfg_.angular_velocity_threshold_deg_s;
            const bool motion_ok =
                motion_energy_norm > (motion_baseline * cfg_.motion_spike_ratio + cfg_.motion_spike_floor);
            const bool age_ok = ps.track_age > cfg_.min_track_age_frames;
            const bool conf_ok = hip_conf > cfg_.min_keypoint_conf;

            if (velocity_ok && angular_ok && motion_ok && age_ok && conf_ok) {
                ps.state = TrackState::Falling;
                ps.down_frame_count = 0;
                ps.still_frame_count = 0;
            }
            break;
        }
        case TrackState::Falling: {
            const bool hip_near_floor =
                has_an && ((ankle.y - hip.y) < cfg_.down_hip_ankle_ratio * ps.baseline_h);
            const bool torso_flat = has_sh && torso_angle > cfg_.down_torso_angle_deg;
            const bool upright_now = has_sh && torso_angle < cfg_.sit_angle_deg;
            const bool still_down = torso_flat || (hip_near_floor && !upright_now);

            if (still_down) {
                ps.down_frame_count++;
            } else {
                ps.state = TrackState::Normal;
                ps.down_frame_count = 0;
                ps.still_frame_count = 0;
                break;
            }

            if (ps.down_frame_count >= cfg_.confirm_frames) {
                if (!cfg_.require_stillness_confirmation) {
                    ps.state = TrackState::Confirmed;
                    just_confirmed = true;
                } else {
                    // Filters floor exercise / push-ups: posture looks
                    // "down" but motion stays high, so stillness never
                    // accumulates and no alert fires.
                    if (motion_energy_norm < cfg_.stillness_motion_threshold) {
                        ps.still_frame_count++;
                    } else {
                        ps.still_frame_count = 0;
                    }
                    if (ps.still_frame_count >= cfg_.stillness_frames) {
                        ps.state = TrackState::Confirmed;
                        just_confirmed = true;
                    }
                }
            }
            break;
        }
        case TrackState::Confirmed: {
            const bool hip_near_floor =
                has_an && ((ankle.y - hip.y) < cfg_.down_hip_ankle_ratio * ps.baseline_h);
            const bool torso_flat = has_sh && torso_angle > cfg_.down_torso_angle_deg;
            const bool upright_now = has_sh && torso_angle < cfg_.sit_angle_deg;
            if (!torso_flat && !(hip_near_floor && !upright_now)) {
                ps.state = TrackState::Normal;
                ps.down_frame_count = 0;
                ps.still_frame_count = 0;
            }
            break;
        }
    }

    if (!just_confirmed) return std::nullopt;

    AlertEvent ev;
    ev.track_id = track_id;
    ev.timestamp_sec = static_cast<double>(frame_idx) / fps;
    ev.reason = "velocity+angular_velocity+motion_spike+stillness confirmed";
    ev.drop_vel_norm = drop_vel_norm;
    ev.angular_vel_deg_s = angular_vel;
    ev.motion_energy = motion_energy_norm;
    ev.torso_angle_deg = torso_angle;
    return ev;
}

void FallStateMachine::evict_stale(std::int64_t frame_idx, std::int64_t max_age_frames) {
    for (auto it = states_.begin(); it != states_.end();) {
        if (frame_idx - it->second.last_frame_idx > max_age_frames) {
            it = states_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace falldet
