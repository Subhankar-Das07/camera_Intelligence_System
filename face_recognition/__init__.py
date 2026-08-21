"""
Face Recognition Engine
=======================
A production-grade, loosely-coupled face recognition module integrated into the
Camera Intelligence System as a standalone pipeline.

Tech Stack:
    - Detection  : SCRFD via InsightFace buffalo_l (ONNX)
    - Recognition: ArcFace via InsightFace buffalo_l (ONNX)
    - Tracking   : ByteTrack via Supervision library
    - Search     : FAISS (CPU, GPU-upgradable)
    - Storage    : Local filesystem (JSON + NPZ + FAISS index)

Hardware: CPU-first. GPU upgrade = swap onnxruntime package only.
"""
