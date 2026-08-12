import sqlite3, sys
from pathlib import Path

db_path = Path("data/smart_vision.db")
print("DB exists:", db_path.exists())
print("DB path:", db_path.resolve())

if db_path.exists():
    size_kb = db_path.stat().st_size / 1024
    print(f"DB size: {size_kb:.1f} KB")
else:
    print("ERROR: No database file found.")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

tables = ["sessions", "reports", "report_objects", "object_events", "tracked_objects", "ocr_events"]
print()
print("=== ROW COUNTS (all sessions) ===")
for t in tables:
    try:
        row = conn.execute("SELECT COUNT(*) as n FROM " + t).fetchone()
        print(f"  {t:<25}: {row['n']} rows")
    except Exception as e:
        print(f"  {t:<25}: ERROR - {e}")

print()
print("=== SESSIONS ===")
sessions = conn.execute(
    "SELECT id, started_at, ended_at, total_reports, total_unique_objects, total_events FROM sessions ORDER BY id DESC LIMIT 5"
).fetchall()
for s in sessions:
    ended = s["ended_at"] or "still running"
    print(f"  Session #{s['id']} | started: {s['started_at']} | ended: {ended} | reports: {s['total_reports']} | objects: {s['total_unique_objects']} | events: {s['total_events']}")

if not sessions:
    print("  No sessions found - backend hasn't run yet!")
    conn.close()
    sys.exit(0)

latest_sid = sessions[0]["id"]
print()
print(f"=== LATEST SESSION #{latest_sid} BREAKDOWN ===")
for t in tables:
    if t == "sessions":
        continue
    try:
        row = conn.execute("SELECT COUNT(*) as n FROM " + t + " WHERE session_id=?", (latest_sid,)).fetchone()
        print(f"  {t:<25}: {row['n']} rows")
    except Exception as e:
        print(f"  {t:<25}: ERROR - {e}")

print()
print("=== RECENT TRACKED OBJECTS (last 10) ===")
objs = conn.execute(
    "SELECT track_id, display_label, category, first_seen, total_duration_sec, highest_confidence, status FROM tracked_objects WHERE session_id=? ORDER BY first_seen DESC LIMIT 10",
    (latest_sid,)
).fetchall()
for o in objs:
    print(f"  #{o['track_id']:<3} {o['display_label']:<20} [{o['category']:<12}] conf:{o['highest_confidence']:.2f} dur:{o['total_duration_sec']:.1f}s status:{o['status']}")
if not objs:
    print("  No tracked objects yet")

print()
print("=== RECENT EVENTS (last 10) ===")
evts = conn.execute(
    "SELECT event_at, event_type, track_id, label, category FROM object_events WHERE session_id=? ORDER BY event_at DESC LIMIT 10",
    (latest_sid,)
).fetchall()
for e in evts:
    print(f"  {e['event_at']} | {e['event_type']:<20} | #{e['track_id']:<3} {e['label']:<20} [{e['category']}]")
if not evts:
    print("  No events yet")

print()
print("=== REPORTS STORED (last 5) ===")
rpts = conn.execute(
    "SELECT id, reported_at, report_number, total_objects, fps, cpu_percent FROM reports WHERE session_id=? ORDER BY reported_at DESC LIMIT 5",
    (latest_sid,)
).fetchall()
for r in rpts:
    print(f"  Report #{r['report_number']:<3} | at: {r['reported_at']} | objects: {r['total_objects']} | fps: {r['fps']} | cpu: {r['cpu_percent']}")
if not rpts:
    print("  No console reports stored yet in this session (reports are written every N seconds by ReportEngine.tick())")

conn.close()
print()
print("VERDICT: DB read SUCCESSFUL. Check counts above.")
