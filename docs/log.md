2026-04-23 | phase-1 | doc | Refined implementation plan into functional-first, slice-based execution checklist.
2026-04-23 | phase-1 | feature | Bootstrapped project structure with src/scripts/tests/notebooks/outputs and dependency baseline.
2026-04-23 | phase-1 | feature | Added IMU and VTS binary parsers with header validation and timestamp monotonicity checks.
2026-04-23 | phase-1 | feature | Implemented frame-to-IMU synchronization utilities and sync delay computation path.
2026-04-23 | phase-1 | feature | Added Phase 1 video pipeline script to render HUD and scrolling IMU telemetry panel.
2026-04-23 | phase-1 | test | Added focused unit test for nearest-sample sync mapping behavior.
2026-04-23 | phase-1 | doc | Added notebook template for binary inspection and README quick-start run command.
2026-04-23 | phase-1 | fix | Adjusted IMU/VTS parser layout assumptions to match real binary structure and validated with 5-frame smoke render.
2026-04-23 | phase-1 | feature | Completed full-length Phase 1 render (1300 frames) to outputs/phase1_imu_sync.mp4.
2026-04-23 | phase-1.1 | polish | Improved HUD contrast/text formatting and plot readability with second-based time axis and clearer labels.
2026-04-23 | phase-1.1 | verify | Rendered polished preview and regenerated full-length polished Phase 1 output video.
2026-04-23 | phase-1.1 | fix | Hardened sync edge handling with signed delay, boundary clamp flags, and minimum IMU length validation.
2026-04-23 | phase-1.1 | feature | Added parser confidence diagnostics and richer timing quality metrics for IMU and VTS streams.
2026-04-23 | phase-1.1 | feature | Added top-10 sync outlier audit printout and unclamped sync stats reporting for clearer alignment quality.
2026-04-23 | phase-1.1 | test | Added sync guardrail unit test and test path bootstrap via tests/conftest.py.
2026-04-23 | phase-1 | verify | Re-ran full Phase 1 pipeline on complete video and regenerated outputs/phase1_imu_sync.mp4 with diagnostics.
2026-04-23 | phase-2 | feature | Added Depth Anything V2 depth module, run_phase2 script, and inferno side-by-side video pipeline.
2026-04-23 | phase-2 | test | Added unit test for depth colormap helper.
2026-04-23 | phase-2 | verify | Smoke-tested Phase 2 on 5 frames; full 1300-frame render saved to outputs/phase2_depth.mp4 (CUDA).
2026-04-23 | phase-3 | feature | Added YOLOv8-seg pipeline, MediaPipe Tasks hand landmarker overlay, and run_phase3 script.
2026-04-23 | phase-3 | fix | Switched hands from deprecated mp.solutions to Hand Landmarker task + bundled model download.
2026-04-23 | phase-3 | test | Added yolo device helper unit tests; ignore models/*.task and yolov8 weights in git.
2026-04-23 | phase-3 | verify | Smoke-tested Phase 3 and completed full 1300-frame render to outputs/phase3_segmentation.mp4.
