# Bundled models

`face_detection_yunet_2023mar.onnx` is the YuNet face detector from the OpenCV model zoo
(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), MIT License,
Copyright (c) 2020 Shiqi Yu. It locates the portrait on passports that print the photo
without a frame, so the photo region can be replaced. It runs through `cv2.FaceDetectorYN`.
