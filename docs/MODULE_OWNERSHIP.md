# Module ownership

| Area | Path | Policy |
|------|------|--------|
| Site Admin (this product UI) | `static/features/site-admin/`, `core/site_admin_*.py`, `core/whatsapp_adapter.py` | Implement on **`develop`** |
| Zone Safety | `static/features/zone-safety/` | Frozen for Site Admin work |
| Vehicle | `static/features/vehicle/` | Frozen |
| Face | `static/features/face/` | Frozen |
| Shell | `static/shell/` | One nav injector + Site Admin highlight only |

Do not create a new git branch for Site Admin unless the team asks. Promote `develop` → `main` when stable.

Teammates continue on their feature branches and must not edit `site-admin/`.
