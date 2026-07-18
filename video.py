import streamlit as st
import tempfile
import os
import cv2


# ==========================================
# VIDEO DETECTION
# ==========================================
def run_video_detection(detector, confidence):

    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is None:
        return

    st.video(uploaded_video)

    if st.button("🎥 Detect Video"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
            input_file.write(uploaded_video.read())
            input_path = input_file.name

        os.makedirs("outputs", exist_ok=True)

        output_path = os.path.join(
            "outputs",
            "detected_video.mp4"
        )

        with st.spinner("🔍 Detecting Objects..."):

            detector.detect_video(
                input_path=input_path,
                output_path=output_path,
                confidence=confidence
            )

        st.success("✅ Video Detection Completed!")

        st.subheader("🎬 Detection Result")

        with open(output_path, "rb") as file:
            video_bytes = file.read()

        st.video(video_bytes)

        st.download_button(
            "⬇ Download Detected Video",
            data=video_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )

        os.remove(input_path)


# ==========================================
# WEBCAM DETECTION
# ==========================================
def run_webcam(detector, confidence):

    st.subheader("📹 Live Webcam Detection")

    st.info(
        "Click Start Webcam to begin live object detection."
    )

    start = st.button(
        "▶ Start Webcam",
        key="start_webcam"
    )

    stop = st.button(
        "⏹ Stop Webcam",
        key="stop_webcam"
    )

    frame_placeholder = st.empty()

    if start:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access webcam.")
            return

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            print("Detector:", detector)
            print("Detector class:", detector.__class__)
            print("Has method:", hasattr(detector, "detect_webcam"))
            print("Methods:", dir(detector))

            annotated_frame, _ = detector.detect_webcam(
                frame,
                confidence
            )

            frame_placeholder.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

            if stop:
                break

        cap.release()

        st.success("✅ Webcam Closed")