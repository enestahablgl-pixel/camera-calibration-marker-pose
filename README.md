# Camera Calibration and STag Marker Pose Estimation

A computer vision project from my internship at Visea İnovatif.
I captured images of a physical STag marker board from different
viewpoints to calibrate a camera and use the calibration for pose estimation.

## Workflow

1. `1_kalibrasyon.py` uses OpenCV to estimate the camera matrix and lens
   distortion coefficients from marker observations. It reports the RMS
   reprojection error and saves the parameters to `kamera_parametreleri.npz`.
2. `2_stag_poz_tahmini.py` loads the saved parameters, a marker map, and
   reference marker coordinates for pose estimation.

## Inputs and outputs

- Calibration takes images of the marker board and generates
  `kamera_parametreleri.npz`.
- Pose estimation requires the generated `.npz` file, `marker_haritasi.txt`,
  and the reference coordinates file configured in `TXT_DOSYASI`.
  ## Demo

![STag marker board used for camera calibration](demo.png)
