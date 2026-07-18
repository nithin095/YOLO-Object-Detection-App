import streamlit as st
import pandas as pd
import numpy as np
import cv2
import time
from PIL import Image
import video
print("VIDEO FILE:", video.__file__)
print("VIDEO MEMBERS:", dir(video))
from detect import YOLODetector
from utils import get_object_counts
from video import run_video_detection, run_webcam

# ----------------------------------
# Page Configuration (Must be first Streamlit command)
# ----------------------------------
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

div[data-testid="stMetric"]{
    background:#20242d;
    padding:20px;
    border-radius:15px;
    border:1px solid #4CAF50;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------
# Load YOLO Model
# ----------------------------------
detector = YOLODetector("models/yolov8n.pt")
import detect

print("Detect module:", detect.__file__)
print("Detector type:", type(detector))
print("Has detect_webcam:", hasattr(detector, "detect_webcam"))
print("Method:", getattr(detector, "detect_webcam", None))
# ----------------------------------
# Main Header
# ----------------------------------
st.title("🤖 AI Object Detection using YOLOv8")

st.caption(
    "Upload an image or video and detect objects in real time using Ultralytics YOLOv8."
)

st.divider()

# ----------------------------------
# Sidebar
# ----------------------------------
with st.sidebar:

    st.title("🤖 AI Vision")

    st.markdown("---")

    st.markdown("### YOLOv8 Object Detection")

    mode = st.radio(
        "Select Detection Mode",
        [
            "Image",
            "Video",
            "Webcam"
        ]
    )

    st.markdown("---")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05
    )

    st.markdown("---")

    st.info("📷 Image Detection")
    st.info("🎥 Video Detection")
    st.info("📹 Webcam (Coming Soon)")

    st.markdown("---")

    st.success("✅ YOLOv8 Model Loaded")

# ==================================
# IMAGE DETECTION
# ==================================

if mode == "Image":

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        if st.button("🚀 Detect Objects"):

            with st.spinner("🔍 Detecting Objects..."):

                start_time = time.time()

                annotated_image, results = detector.detect(
                    image_np,
                    confidence
                )

                end_time = time.time()

                processing_time = end_time - start_time

            with col2:
                st.subheader("🎯 Detection Result")

                st.image(
                    annotated_image,
                    use_container_width=True
                )

            counts = get_object_counts(results)

            total_objects = sum(counts.values())

            unique_objects = len(counts)

            st.divider()

            st.markdown("## 📊 Detection Dashboard")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="👤 Objects Detected",
                    value=total_objects
                )

            with col2:
                st.metric(
                    label="📦 Unique Classes",
                    value=unique_objects
                )

            with col3:
                st.metric(
                    label="🎯 Confidence",
                    value=f"{confidence:.2f}"
                )

            st.divider()

            st.subheader("📊 Detection Summary")

            df = pd.DataFrame(
                counts.items(),
                columns=[
                    "Object",
                    "Count"
                ]
            )

            st.dataframe(
                df,
                use_container_width=True
            )
            
            import plotly.express as px

            st.subheader("🥧 Object Distribution")

            fig = px.pie(
                df,
                values="Count",
                names="Object",
                hole=0.4,
                title="Detected Objects"
            )

            fig.update_traces(textposition="inside", textinfo="percent+label")

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            
            st.subheader("📊 Object Count Analysis")

            fig_bar = px.bar(
                df,
                x="Object",
                y="Count",
                color="Object",
                text="Count",
                title="Detected Object Counts"
            )

            fig_bar.update_traces(
                textposition="outside"
            )

            fig_bar.update_layout(
                xaxis_title="Object",
                yaxis_title="Count",
                showlegend=False
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )
            success, buffer = cv2.imencode(
                ".jpg",
                annotated_image
            )

            if success:

                st.download_button(
                    "⬇ Download Detection Image",
                    buffer.tobytes(),
                    file_name="Detection.jpg",
                    mime="image/jpeg"
                )
            
            st.subheader("📋 Detection Details")

            details = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    details.append({
                        "Object": result.names[int(box.cls)],
                        "Confidence": round(float(box.conf), 2),
                        "X": int(x1),
                        "Y": int(y1),
                        "Width": int(x2 - x1),
                        "Height": int(y2 - y1)
                    })
            
            details_df = pd.DataFrame(details)

            st.dataframe(
                details_df,
                use_container_width=True
            )
            csv = details_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📄 Download Detection Report (CSV)",
                data=csv,
                file_name="detection_report.csv",
                mime="text/csv"
            )
            st.success("✅ Detection Completed Successfully!")

            st.info(
                f"⏱ Processing Time : {processing_time:.2f} seconds"
            )
# ==================================
# VIDEO DETECTION
# ==================================

elif mode == "Video":
    
    run_video_detection(
        detector,
        confidence
    )

elif mode == "Webcam":
    run_webcam(
        detector,
        confidence
    )
    
st.divider()

st.markdown(
    """
    <center>
        Developed with ❤️ using Python, Streamlit & YOLOv8
    </center>
    """,
    unsafe_allow_html=True
)