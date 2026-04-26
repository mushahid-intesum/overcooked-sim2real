"""
overcooked_server.py
=====================
FastAPI server for the Overcooked environment data collector.

Endpoints:
  GET  /layout         → tell the robot which layout is active
  POST /upload         → receive a JPEG from the robot and save it
  POST /endscan        → robot finished one position; return next move command

Image save path:
  images/<layout>/step_<step>_<dir>_<shot>.jpg
  e.g.  images/kitchen_A/step_02_N_3.jpg

Layout-to-route mapping:
  Edit LAYOUT_ROUTES below to define the movement sequence for each layout.
  Each entry is a list of cardinal directions the robot should step through.
  The server tracks the position index and returns the next move, or "DONE"
  when the route is exhausted.

Install:
  pip install fastapi uvicorn python-multipart

Run:
  uvicorn overcooked_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Active layout (change this to switch layouts) ───────────────────────────
ACTIVE_LAYOUT = "kitchen_A"

# ─── Route definitions ────────────────────────────────────────────────────────
# For each layout, define the ordered sequence of moves the robot should make
# after completing each scan position. The robot starts at position 0 (no move),
# scans, then receives LAYOUT_ROUTES[layout][0] to move to position 1, etc.
# "DONE" is returned automatically after the last position is scanned.
#
# Example grid for kitchen_A (3x3 grid, row-major, moving East then South):
#   [0,0] → E → [0,1] → E → [0,2] → S → [1,2] → W → [1,1] → W → [1,0] → ...
LAYOUT_ROUTES: dict[str, list[str]] = {
    "kitchen_A": ["E", "S", "W", "S", "E"],   # 3×3 snake
    "kitchen_B": ["E", "S", "W", "S", "E"],                   # L-shaped
    "kitchen_C": ["N", "E", "S", "S", "W", "N"],              # custom
}

# ─── Image output root ────────────────────────────────────────────────────────
IMAGE_ROOT = Path("images")

# ─── In-memory scan log (per session) ────────────────────────────────────────
scan_log: list[dict] = []

app = FastAPI(title="Overcooked Data Collector")


# ─────────────────────────────────────────────────────────────────────────────
#  GET /layout
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/layout")
def get_layout():
    """Return the currently active layout name to the robot."""
    return {"layout": ACTIVE_LAYOUT}


# ─────────────────────────────────────────────────────────────────────────────
#  POST /upload
#  Form fields: layout, step (int), dir (N/E/S/W), shot (0-4), file (JPEG)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_image(
    layout: str      = Form(...),
    step:   int      = Form(...),
    dir:    str      = Form(...),
    shot:   int      = Form(...),
    file:   UploadFile = File(...),
):
    # Sanitize inputs
    layout = layout.strip().replace("/", "_").replace("..", "")
    dir    = dir.strip().upper()
    if dir not in ("N", "E", "S", "W"):
        raise HTTPException(status_code=400, detail=f"Invalid direction: {dir}")
    if not (0 <= shot < 10):
        raise HTTPException(status_code=400, detail=f"Invalid shot index: {shot}")

    # Build output folder:  images/<layout>/
    out_dir = IMAGE_ROOT / layout
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename:  step_02_N_3.jpg
    filename = f"step_{step:03d}_{dir}_{shot}.jpg"
    out_path = out_dir / filename

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    out_path.write_bytes(contents)
    size_kb = len(contents) / 1024

    print(f"[UPLOAD] {layout}/{filename}  ({size_kb:.1f} KB)")
    return {"status": "ok", "saved": str(out_path)}


# ─────────────────────────────────────────────────────────────────────────────
#  POST /endscan
#  Body: {"layout": "kitchen_A", "step": 2}
#  Returns: {"move": "E"} or {"move": "DONE"}
# ─────────────────────────────────────────────────────────────────────────────
class EndScanRequest(BaseModel):
    layout: str
    step:   int

@app.post("/endscan")
def end_scan(req: EndScanRequest):
    layout = req.layout.strip()
    step   = req.step

    # Log this completed scan position
    scan_log.append({
        "layout":    layout,
        "step":      step,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"[ENDSCAN] layout={layout}  step={step}  total_scanned={len(scan_log)}")

    # Look up route
    route = LAYOUT_ROUTES.get(layout)
    if route is None:
        print(f"[WARN] Unknown layout '{layout}', sending DONE")
        return {"move": "DONE"}

    # step N was just scanned; the robot needs to move to reach step N+1
    # move index = step (0-indexed)
    if step < len(route):
        next_move = route[step]
        print(f"[ENDSCAN] → move={next_move}")
        return {"move": next_move}
    else:
        print("[ENDSCAN] Route complete → DONE")
        _save_scan_log(layout)
        return {"move": "DONE"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_scan_log(layout: str):
    """Persist the scan log to JSON when collection finishes."""
    log_path = IMAGE_ROOT / layout / "scan_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(scan_log, f, indent=2)
    print(f"[LOG] Scan log saved to {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick status endpoint (optional, useful for debugging)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    """Return a summary of images collected so far."""
    summary = {}
    if IMAGE_ROOT.exists():
        for layout_dir in IMAGE_ROOT.iterdir():
            if layout_dir.is_dir():
                jpegs = list(layout_dir.glob("*.jpg"))
                summary[layout_dir.name] = len(jpegs)
    return {
        "active_layout": ACTIVE_LAYOUT,
        "images_collected": summary,
        "scan_positions_logged": len(scan_log),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("overcooked_server:app", host="0.0.0.0", port=8000, reload=True)