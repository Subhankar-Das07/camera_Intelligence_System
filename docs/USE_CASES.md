# Use cases

Rules are the heart of Site Admin. Each saved rule is `camera + scan_type + geometry (when needed) + channels`. Monitor, Alerts, and Reports run on top of those rules.

## Available now

| Label | `scan_type` | Typical need | Fit | Notes |
|-------|-------------|--------------|-----|-------|
| Intrusion / restricted area | `intrusion` | Person enters a drawn area | High | ROI required |
| Danger zone | `danger_zone` | Person near machines / hazards | High | ROI required |
| Fall detection | `fall` | Person falls | High | ROI optional |
| Gate / entrance analytics | `gate_analytics` | Count line, zones, open/close | High | Gate geometry required |
| Face / staff attendance | `face_attendance` | Staff presence; approve in **Staff** | High | Identity — learn then approve |
| Vehicle / number plate | `vehicle` | Plates; approve in **Vehicles** | High | See identity flow below |

## Coming soon (listed in Rules dropdown; cannot save yet)

| Label | `scan_type` | Typical need | Fit |
|-------|-------------|--------------|-----|
| Loitering | `loitering` | Person stays in area too long | High — shops, alleys |
| Line crossing / tripwire | `line_crossing` | Cross a line either way | High — simpler than full gate |
| Crowding / occupancy | `crowding` | Too many people in a zone | High — queues |
| Object left / abandoned | `object_left` | Bag left behind | Medium — shops, lobbies |
| Object removed | `object_removed` | Watched item missing | Medium — displays |
| Fire / smoke | `fire_smoke` | Smoke or fire | High for safety |
| Shoplifting / unusual motion | `shoplifting` | Suspicious retail motion | Medium — retail |
| PPE (helmet / vest) | `ppe` | Missing PPE | High — factories |
| Wrong-way / vehicle direction | `wrong_way` | Vehicle wrong direction | Medium — one-way |
| Animal / pet | `animal_pet` | Pet in area (counts in Reports later) | Medium — homes |

## Identity flow (Staff / Vehicles)

1. Save a Face or Vehicle rule and let it run.
2. Candidates appear in **Staff** or **Vehicles** (thumbs + Approve / Ignore).
3. Approved = “ours”; later, unknowns can trigger alerts.

### Vehicle plates (dedupe + risk)

1. First OCR of a plate → one **Vehicles** row (thumb) + one **unknown** alert (not repeated every frame).
2. **Approve (ours)** → society vehicle; later sightings are silent (last-seen still updates).
3. **Ignore** → silent on return.
4. **Mark risk** → next time that plate appears, **immediate high alert** (not suppressed by normal rule cooldown).
5. Same normalized plate is never duplicated in the gallery.

Fire and shoplifting remain Coming soon pipelines (not separate engineer-tab rewrites).
