from ultralytics import YOLO
import cv2


class YOLODetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # -----------------------
    # Image Detection
    # -----------------------
    def detect(self, image, confidence=0.25):

        results = self.model.predict(
            source=image,
            conf=confidence,
            save=False,
            verbose=False
        )

        annotated = results[0].plot()

        return annotated, results

    # -----------------------
    # Video Detection
    # -----------------------
    def detect_video(self, input_path, output_path, confidence=0.25):

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise Exception("Cannot open video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            writer.write(annotated)

        cap.release()
        writer.release()

    # -----------------------
    # Webcam Detection
    # -----------------------
    def detect_webcam(self, frame, confidence=0.25):

        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False
        )

        annotated = results[0].plot()

        return annotated, results