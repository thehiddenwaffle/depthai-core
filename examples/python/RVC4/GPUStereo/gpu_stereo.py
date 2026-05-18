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
# GPUStereo follows the same pattern as StereoDepth: configure via initialConfig.
# Note: only `confidenceThreshold` is supported for GPUStereoConfig.
gpu.initialConfig.confidenceThreshold = 25
conf_q = gpu.confidenceMap.createOutputQueue()

disp_q = gpu.disparity.createOutputQueue()

with pipeline:
    pipeline.start()
    while pipeline.isRunning():
        frame = disp_q.get()
        disp = frame.getFrame()
        conf = conf_q.get()

        disp8 = (disp.astype(np.float32) * (255.0 / 96.0)).clip(0, 255).astype(np.uint8)
        disp8 = cv2.applyColorMap(disp8, cv2.COLORMAP_JET)

        c = conf.getFrame()  # RAW8 confidence map
        cv2.imshow("GPUStereo Disparity", disp8)
        cv2.imshow("GPUStereo Confidence", c)
        if cv2.waitKey(1) == ord("q"):
            break
