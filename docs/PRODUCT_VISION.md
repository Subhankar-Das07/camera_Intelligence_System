# Product vision

Camera Intelligence is an **edge** product for shops, homes, and small factories that already have CCTV.

A mini-PC on the same LAN as the NVR runs Docker. It reads RTSP (or uploaded test video) locally, applies **rules** the Admin set once, and produces **alerts** and **reports**. Raw video is not sent to a central cloud.

**Client surface:** Site Admin tab only (`/features/site-admin/`).

**Engineer surface:** existing Zone Safety, Vehicle, and Face tabs — do not rewrite those for product work.
