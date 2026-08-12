"""
report_engine.py
================
Structured console scene reporter for the Smart Vision Assistant.

Produces numbered per-object analysis reports like:

  +============================================================+
  |              SCENE REPORT  [2026-07-10 15:55:01]           |
  +============================================================+
  |  OBJECT 1 — Person  (Humans)                               |
  |    Track: #3  |  Confidence: 87%  |  In scene: 34s         |
  |    ▸ Wearing a dark blue shirt                             |
  |    ▸ Seated, facing camera                                 |
  |    ▸ Approx. adult male                                    |
  +------------------------------------------------------------+
  |  OBJECT 2 — Laptop  (Electronics)                          |
  |    Track: #7  |  Confidence: 91%  |  In scene: 1m 12s      |
  |    ▸ Silver metallic body                                  |
  |    ▸ Screen appears to be on                               |
  |    ▸ Open lid, keyboard visible                            |
  +------------------------------------------------------------+
  |  OBJECT 3 — Bottle  (Kitchen)  [analysing...]              |
  |    Track: #2  |  Confidence: 63%  |  In scene: 8s           |
  |    ▸ Pending Gemini analysis...                            |
  +============================================================+
  |  Total: 3 objects  |  NEW: 1  |  REMOVED: 0               |
  |  Scene Stability   : 74.0%                                 |
  |  System            : FPS 22.4 | CPU 38% | RAM 487 MB      |
  +============================================================+
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from scene_memory import CATEGORY_ORDER, ObjectRecord, SceneMemory

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional: psutil for system stats
# ──────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    log.debug("psutil not installed -- system stats (CPU/RAM) will be unavailable.")


class ReportEngine:
    """
    Generates and prints structured per-object scene reports to the console.
    Optionally writes reports to a log file.

    Call report_engine.tick(fps) once per frame; it will print a report
    automatically when REPORT_INTERVAL seconds have elapsed.
    """

    _BOX_WIDTH = 62  # Inner character width of the report box

    def __init__(self, scene_memory: SceneMemory, db=None, session_id: Optional[int] = None) -> None:
        self._memory = scene_memory
        self._db = db                  # DatabaseManager instance (optional, None = no DB)
        self._session_id = session_id  # Current session ID for DB writes
        self._last_report_time: float = time.monotonic()
        self._report_count: int = 0
        self._current_fps: float = 0.0
        self._first_report_done: bool = False          # Track if first report has fired
        self._start_time: float = time.monotonic()     # App start time
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._file_handle = None

        if config.REPORT_TO_FILE:
            self._init_file_logging()

    # ──────────────────────────────────────────────────────────────────────────
    # File Logging
    # ──────────────────────────────────────────────────────────────────────────

    def _init_file_logging(self) -> None:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = reports_dir / f"session_{timestamp}.log"
        try:
            self._file_handle = open(log_path, "w", encoding="utf-8")
            log.info("Scene reports will be saved to: %s", log_path)
        except OSError as exc:
            log.warning("Could not open report file: %s", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # Tick (called every frame)
    # ──────────────────────────────────────────────────────────────────────────

    def tick(self, fps: float) -> bool:
        """
        Update current FPS and check if a report is due.
        Returns True if a report was printed this call.

        First report fires 5 seconds after startup (not after a full interval)
        so the user immediately sees console output when objects appear.
        """
        self._current_fps = fps
        now = time.monotonic()

        # Fire an early first report 5 seconds after startup
        if not self._first_report_done and (now - self._start_time) >= 5.0:
            if self._memory.active_count() > 0:   # Only if objects are visible
                self._first_report_done = True
                self._last_report_time = now
                self._report_count += 1
                self._print_report()
                return True

        if now - self._last_report_time >= config.REPORT_INTERVAL:
            self._first_report_done = True
            self._last_report_time = now
            self._report_count += 1
            self._print_report()
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Report Assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _print_report(self) -> None:
        lines, report_data = self._build_report_data()
        output = "\n".join(lines)
        # flush=True ensures it appears immediately in Windows console (no buffering)
        print(output, flush=True)
        sys.stdout.flush()
        if self._file_handle:
            try:
                self._file_handle.write(output + "\n")
                self._file_handle.flush()
            except OSError:
                pass
        # Persist to database if connected
        if self._db is not None and self._session_id is not None:
            self._save_to_db(report_data)

    def _save_to_db(self, report_data: dict) -> None:
        """Write this report and all its object rows to the database."""
        try:
            stats = report_data["system_stats"]
            report_id = self._db.save_report(
                session_id=self._session_id,
                report_number=self._report_count,
                total_objects=report_data["total_objects"],
                new_objects=report_data["new_objects"],
                removed_objects=report_data["removed_objects"],
                scene_stability=report_data["scene_stability"],
                fps=stats.get("fps"),
                cpu_percent=stats.get("cpu"),
                ram_mb=stats.get("ram_mb"),
            )
            # save_report_objects is async (fire-and-forget)
            self._db.save_report_objects(
                report_id, self._session_id, report_data["objects"]
            )
            log.debug("[DB] Report #%d saved (id=%d).", self._report_count, report_id)
        except Exception as exc:
            log.warning("[DB] Failed to save report: %s", exc)

    def _build_report_data(self):
        """
        Build the report lines AND collect all data as a dict for DB persistence.
        Returns (lines, report_data_dict).
        """
        W = self._BOX_WIDTH
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append(self._top())
        lines.append(self._title(f"SCENE REPORT  [{now_str}]", W))
        lines.append(self._top())

        # ── Collect all active objects sorted by first_seen (oldest first) ────
        all_records: List[ObjectRecord] = self._memory.get_active_records()
        all_records.sort(key=lambda r: r.first_seen)

        new_count = sum(1 for r in all_records if r.is_new)
        removed_count = len(self._memory.get_recently_removed())
        stability = self._memory.get_stability_pct()
        stats = self._get_system_stats()

        # Build object rows list for DB
        db_objects = []
        if not all_records:
            lines.append(self._padded("  No objects currently detected.", W))
        else:
            for idx, rec in enumerate(all_records, start=1):
                lines.extend(self._object_block(idx, rec, W))
                if idx < len(all_records):
                    lines.append(self._divider())
                db_objects.append({
                    "track_id":           rec.track_id,
                    "object_number":      idx,
                    "label":              rec.display_label,
                    "yolo_label":         rec.yolo_label,
                    "category":           rec.category,
                    "confidence":         rec.confidence,
                    "duration_seconds":   rec.duration,
                    "is_new":             rec.is_new,
                    "is_gemini_verified": rec.gemini_verified,
                    "gemini_description": rec.gemini_description,
                    # Bounding box coordinates for spatial analytics
                    "bbox_x1": rec.bbox[0] if rec.bbox else None,
                    "bbox_y1": rec.bbox[1] if rec.bbox else None,
                    "bbox_x2": rec.bbox[2] if rec.bbox else None,
                    "bbox_y2": rec.bbox[3] if rec.bbox else None,
                })

        # ── Footer ────────────────────────────────────────────────────────────
        lines.append(self._top())
        total_str = f"Total: {len(all_records)} object{'s' if len(all_records) != 1 else ''}"
        footer_counts = f"{total_str}  |  NEW: {new_count}  |  REMOVED: {removed_count}"
        lines.append(self._padded(f"  {footer_counts}", W))
        lines.append(self._kv_row("Scene Stability", f"{stability:.1f}%", W))
        lines.append(self._kv_row("System", self._system_stats_str(stats), W))
        lines.append(self._bottom())

        report_data = {
            "total_objects":  len(all_records),
            "new_objects":    new_count,
            "removed_objects": removed_count,
            "scene_stability": stability,
            "system_stats":   stats,
            "objects":        db_objects,
        }
        return lines, report_data

    def _object_block(self, idx: int, rec: ObjectRecord, W: int) -> List[str]:
        """Build the lines for a single numbered object entry."""
        lines = []

        # ── Header line: OBJECT N — Label  (Category)  [status tag] ──────────
        status_tag = ""
        if rec.is_new:
            status_tag = "  [NEW]"
        elif config.ENABLE_GEMINI:
            if rec.gemini_verified:
                status_tag = "  [Gemini verified]"
            elif rec.gemini_description:
                status_tag = "  [analysed]"
            elif rec.gemini_skipped_reason == "LOW_FPS":
                status_tag = "  [Skipped: Low FPS]"
            elif rec.gemini_skipped_reason in ("BUDGET_EXCEEDED", "GEMINI_BUDGET_EXCEEDED"):
                status_tag = "  [Skipped: Budget Exceeded]"
            elif rec.gemini_skipped_reason == "QUEUE_FULL":
                status_tag = "  [Skipped: Queue Full]"
            elif rec.gemini_skipped_reason == "GEMINI_UNAVAILABLE":
                status_tag = "  [Failed: API Unavailable]"
            elif rec.gemini_skipped_reason == "GEMINI_TIMEOUT":
                status_tag = "  [Failed: Timeout]"
            elif rec.gemini_skipped_reason == "GEMINI_API_ERROR":
                status_tag = "  [Failed: API Error]"
            elif rec.gemini_skipped_reason == "GEMINI_PARSE_FAILED":
                status_tag = "  [Failed: Parse Error]"
            else:
                status_tag = "  [analysing...]"

        header = f"  OBJECT {idx} — {rec.display_label.title()}  ({rec.category}){status_tag}"
        lines.append(self._padded(header, W))

        # ── Detail line: track, confidence, duration ────────────────────────
        detail = (
            f"    Track: #{rec.track_id}"
            f"  |  Confidence: {rec.confidence:.0%}"
            f"  |  In scene: {rec.duration_str()}"
        )
        lines.append(self._padded(detail, W))

        # ── Gemini description bullets ─────────────────────────────────────
        if config.ENABLE_GEMINI:
            if rec.gemini_description:
                for bullet_line in rec.gemini_description.splitlines():
                    lines.append(self._padded(bullet_line, W))
            else:
                if rec.gemini_skipped_reason == "LOW_FPS":
                    lines.append(self._padded("    ▸ [Skipped: Low FPS]", W))
                elif rec.gemini_skipped_reason in ("BUDGET_EXCEEDED", "GEMINI_BUDGET_EXCEEDED"):
                    lines.append(self._padded("    ▸ [Skipped: Budget Exceeded]", W))
                elif rec.gemini_skipped_reason == "QUEUE_FULL":
                    lines.append(self._padded("    ▸ [Skipped: Queue Full]", W))
                elif rec.gemini_skipped_reason == "GEMINI_UNAVAILABLE":
                    lines.append(self._padded("    ▸ [Failed: API Unavailable]", W))
                elif rec.gemini_skipped_reason == "GEMINI_TIMEOUT":
                    lines.append(self._padded("    ▸ [Failed: Timeout]", W))
                elif rec.gemini_skipped_reason == "GEMINI_API_ERROR":
                    lines.append(self._padded("    ▸ [Failed: API Error]", W))
                elif rec.gemini_skipped_reason == "GEMINI_PARSE_FAILED":
                    lines.append(self._padded("    ▸ [Failed: Parse Error]", W))
                elif rec.is_new:
                    lines.append(self._padded("    ▸ Just appeared — analysis queued...", W))
                else:
                    lines.append(self._padded("    ▸ Pending Gemini analysis...", W))

        return lines

    def _get_system_stats(self) -> dict:
        """Return system stats as a dict (for DB storage and display)."""
        stats = {"fps": self._current_fps, "cpu": None, "ram_mb": None}
        if _PSUTIL_AVAILABLE and self._process:
            try:
                stats["cpu"] = psutil.cpu_percent(interval=None)
                stats["ram_mb"] = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return stats

    def _system_stats_str(self, stats: dict) -> str:
        """Format system stats dict as a display string."""
        fps_str = f"FPS {stats['fps']:.1f}"
        if stats["cpu"] is not None and stats["ram_mb"] is not None:
            return f"{fps_str}  |  CPU {stats['cpu']:.0f}%  |  RAM {stats['ram_mb']:.0f} MB"
        return f"{fps_str}  |  CPU N/A  |  RAM N/A"


    # ──────────────────────────────────────────────────────────────────────────
    # Box Drawing Helpers (ASCII -- compatible with all Windows terminals)
    # ──────────────────────────────────────────────────────────────────────────

    def _top(self) -> str:
        return "+" + "=" * (self._BOX_WIDTH + 2) + "+"

    def _bottom(self) -> str:
        return "+" + "=" * (self._BOX_WIDTH + 2) + "+"

    def _divider(self) -> str:
        return "+" + "-" * (self._BOX_WIDTH + 2) + "+"

    def _title(self, text: str, width: int) -> str:
        return "|  " + text.center(width) + "  |"

    def _padded(self, text: str, width: int) -> str:
        """Pad a raw text line into the box, truncating if needed."""
        if len(text) > width:
            text = text[:width - 3] + "..."
        return f"|{text:<{width + 2}}|"

    def _kv_row(self, key: str, value: str, width: int) -> str:
        """Format a key: value footer row."""
        key_col = f"{key:<18}"
        available = width - len(key_col) - 4
        if len(value) > available:
            value = value[:available - 3] + "..."
        return f"|  {key_col}: {value:<{available}}  |"

    # ──────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._db and self._session_id:
            try:
                summary = self._db.get_session_summary(self._session_id)
                W = self._BOX_WIDTH
                
                lines = []
                lines.append(self._top())
                lines.append(self._title("SESSION ANALYTICS SUMMARY", W))
                lines.append(self._top())
                
                lines.append(self._padded("  Most Frequent Objects:", W))
                for idx, obj in enumerate(summary.get("most_common", []), 1):
                    lines.append(self._padded(f"    {idx}. {obj['label']} ({obj['count']} occurrences)", W))
                    
                lines.append(self._padded("  Longest Dwell Times:", W))
                for idx, obj in enumerate(summary.get("longest_dwell", []), 1):
                    lines.append(self._padded(f"    {idx}. {obj['label']} ({obj['duration']:.1f}s)", W))
                
                lines.append(self._padded("  Avg Dwell by Category (Completed):", W))
                for idx, cat in enumerate(summary.get("avg_dwell_by_category", []), 1):
                    lines.append(self._padded(f"    {idx}. {cat['category']} ({cat['duration']:.1f}s)", W))
                
                lines.append(self._divider())
                lines.append(self._padded(f"  Total Scene Events Logged: {summary.get('total_events', 0)}", W))
                lines.append(self._bottom())
                
                output = "\n".join(lines)
                print("\n" + output, flush=True)
                
                if self._file_handle:
                    self._file_handle.write("\n" + output + "\n")
                    self._file_handle.flush()
            except Exception as e:
                log.warning("Could not generate session summary: %s", e)

        if self._file_handle:
            try:
                self._file_handle.close()
            except OSError:
                pass
        log.info(
            "[Report] Session complete. Total reports generated: %d",
            self._report_count,
        )
