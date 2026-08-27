"""FastAPI routes for search replay under /api/site-admin/search."""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from core import site_admin_search as search_engine
from core import site_admin_search_store as sstore

router = APIRouter(prefix="/api/site-admin/search", tags=["site-admin-search"])

os.makedirs(sstore.SEARCH_THUMB_DIR, exist_ok=True)


def _require_admin(x_cis_role: Optional[str]) -> None:
    role = (x_cis_role or "admin").strip().lower()
    if role == "viewer":
        raise HTTPException(status_code=403, detail="Admin only")


class SearchJobIn(BaseModel):
    camera_id: str
    rule_ids: List[str] = Field(default_factory=list)
    start_sec: float = 0.0
    end_sec: Optional[float] = None


@router.get("/cameras")
def list_file_cameras(x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    return {"cameras": sstore.list_file_cameras()}


@router.get("/cameras/{camera_id}/video-meta")
def get_video_meta(camera_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    from core import site_admin_store as store

    if not store.get_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        return {"meta": sstore.get_video_meta(camera_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/cameras/{camera_id}/rules")
def list_camera_rules(camera_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    from core import site_admin_store as store

    if not store.get_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"rules": sstore.list_camera_rules(camera_id)}


@router.post("/jobs")
def create_job(body: SearchJobIn, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    try:
        job = sstore.create_job(body.camera_id, body.rule_ids, body.start_sec, body.end_sec)
        search_engine.spawn_search_job(job["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"job": job}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    job = sstore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job, "progress": sstore.get_progress(job_id)}


@router.get("/jobs/{job_id}/results")
def get_results(job_id: str, x_cis_role: Optional[str] = Header(default="admin")):
    _require_admin(x_cis_role)
    if not sstore.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"results": sstore.get_results(job_id)}
