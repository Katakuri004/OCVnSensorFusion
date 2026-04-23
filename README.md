# Trekion Technical Assignment

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run Phase 1 pipeline:
   - `python scripts/run_phase1.py --imu data/recording2.imu --vts data/recording2.vts --video data/recording2.mp4 --output outputs/phase1_imu_sync.mp4`
4. Run Phase 2 depth pipeline (downloads model on first run; GPU recommended):
   - `python scripts/run_phase2.py --video data/recording2.mp4 --output outputs/phase2_depth.mp4`
   - Optional: `--device cpu` or `--device cuda`
5. Run Phase 3 segmentation (downloads YOLO weights on first run):
   - `python scripts/run_phase3.py --video data/recording2.mp4 --output outputs/phase3_segmentation.mp4`
   - Optional: `--model yolov8s-seg.pt`, `--no-hands`, `--device cuda`

## Project structure

- `src/trekion/` - reusable parsing, synchronization, and visualization modules
- `scripts/` - executable entry points
- `tests/` - focused unit tests
- `notebooks/` - notebook-first exploration and validation
- `docs/` - assignment docs, implementation plan, and log
- `outputs/` - generated deliverable videos
