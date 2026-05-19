#!/usr/bin/env python3
"""Minimal GPUStereo example — disparity from a stereo camera pair (RVC4 only)."""

import cv2
import depthai as dai
import numpy as np

device = dai.Device()

if not device.isGpuAvailable():
    print("Exiting GPUStereo example: GPU not available on this device.")
    raise SystemExit(0)

pipeline = dai.Pipeline(device)
cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

gpu = pipeline.create(dai.node.GPUStereo)
cam_left.requestOutput((1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=30).link(gpu.left)
cam_right.requestOutput((1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=30).link(gpu.right)
gpu.setRectification(True)
gpu.setConfidenceThreshold(25)

disp_q = gpu.disparity.createOutputQueue()

with pipeline:
    pipeline.start()
    while pipeline.isRunning():
        frame = disp_q.get()
        disp = frame.getFrame()

        disp8 = (disp.astype(np.float32) * (255.0 / 96.0)).clip(0, 255).astype(np.uint8)
        disp8 = cv2.applyColorMap(disp8, cv2.COLORMAP_JET)

        cv2.imshow("GPUStereo Disparity", disp8)
        if cv2.waitKey(1) == ord("q"):
            break
