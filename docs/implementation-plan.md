# Trekion Implementation Plan

## Working style
- Build in small, testable slices.
- Prioritize functional correctness first, then optimization.
- Keep code clean and modular with minimal comments.
- Append one-line progress entries to `docs/log.md`.

## Phase 1 (done): Parsing + synchronization + IMU sync video
### Goal
Ship a fully functional end-to-end pipeline for Task 1 before moving to depth and segmentation.

### Deliverable
- `outputs/phase1_imu_sync.mp4` containing:
  - camera frame
  - scrolling IMU plots (acc/gyro/mag, XYZ)
  - HUD with frame number, timestamps, current values, IMU Hz, camera FPS, temperature, and sync delay stats

### Implementation slices
1. **Scaffold**
   - Create `src/`, `scripts/`, `tests/`, `notebooks/`, `outputs/`.
   - Add dependency file and README run instructions.
2. **Binary parser**
   - Parse `.imu` with `TRIMU001` validation.
   - Parse `.vts` with `TRIVTS01` validation.
   - Validate record-size assumptions and monotonic timestamps.
3. **Synchronization**
   - Map each frame timestamp from `.vts` to nearest IMU sample.
   - Provide window selection for scrolling plot.
   - Compute sync delay stats (mean, median, max).
4. **Renderer**
   - Read video frames from `data/recording2.mp4`.
   - Render IMU plots into buffer and compose side-by-side frame.
   - Draw required HUD fields.
5. **Validation**
   - Add focused tests for sync math.
   - Run a short clip first, then full render.

### Exit criteria
- Parsers run cleanly on files in `data/`.
- Sync statistics are computed and visible in HUD.
- Phase 1 output video is generated successfully.

## Phase 2 (done): Dense depth video
- `scripts/run_phase2.py` uses Depth Anything V2 Small (HF) by default.
- Side-by-side: left RGB, right depth with `COLORMAP_INFERNO`, resized to frame size.
- Export `outputs/phase2_depth.mp4`. Fisheye: raw frames unless you add undistort later.

## Phase 3 (done): Detection/segmentation video
- `scripts/run_phase3.py`: Ultralytics YOLOv8-seg (`yolov8n-seg.pt` default), `plot()` overlays + optional MediaPipe Hand Landmarker (Tasks API).
- Export `outputs/phase3_segmentation.mp4`. First run downloads `models/hand_landmarker.task`.

## Phase 4: Final docs and polish
- Final README and setup reproducibility checks.
- Brief write-up for parsing approach, model choices, and trade-offs.
- Final review of logs and deliverables.