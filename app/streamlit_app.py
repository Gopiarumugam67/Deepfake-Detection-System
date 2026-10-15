"""
Streamlit UI for the Deepfake Media Detection System.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CFG  # noqa: E402
from src.inference.image_predictor import ImagePredictor  # noqa: E402
from src.inference.video_predictor import VideoPredictor  # noqa: E402

st.set_page_config(page_title="Deepfake Media Detector", page_icon="🕵️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_image_predictor() -> ImagePredictor:
    return ImagePredictor()


@st.cache_resource(show_spinner="Loading model...")
def load_video_predictor() -> VideoPredictor:
    return VideoPredictor()


def render_verdict(label: str, confidence: float) -> None:
    if label == "real":
        st.success(f"✅ Prediction: REAL  (confidence: {confidence:.1%})")
    else:
        st.error(f"⚠️ Prediction: FAKE  (confidence: {confidence:.1%})")
    st.progress(confidence)


def main() -> None:
    st.title("🕵️ Deepfake Media Detector")
    st.caption(
        f"Model: {CFG.train.model_name} | Trained to classify faces as Real or Deepfake"
    )

    tab_image, tab_video, tab_about = st.tabs(["Image", "Video", "About"])

    with tab_image:
        st.subheader("Upload an image")
        uploaded_image = st.file_uploader(
            "JPG or PNG containing a visible face", type=["jpg", "jpeg", "png"], key="image_uploader"
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

            if st.button("Analyze image", type="primary"):
                try:
                    predictor = load_image_predictor()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix) as tmp:
                    tmp.write(uploaded_image.getvalue())
                    tmp_path = Path(tmp.name)

                with st.spinner("Detecting face and running classifier..."):
                    result = predictor.predict(tmp_path)

                if not result["face_detected"]:
                    st.warning(result["error"])
                else:
                    render_verdict(result["label"], result["confidence"])
                    st.caption(f"Raw fake probability: {result['fake_probability']:.2%}")

    with tab_video:
        st.subheader("Upload a video")
        uploaded_video = st.file_uploader(
            "MP4/MOV/AVI containing a visible face", type=["mp4", "mov", "avi"], key="video_uploader"
        )
        col1, col2 = st.columns(2)
        frame_stride = col1.number_input("Frame stride", min_value=1, max_value=60, value=10)
        max_frames = col2.number_input("Max frames to sample", min_value=1, max_value=100, value=30)

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("Analyze video", type="primary"):
                try:
                    predictor = load_video_predictor()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
                predictor.frame_stride = int(frame_stride)
                predictor.max_frames = int(max_frames)

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
                    tmp.write(uploaded_video.getvalue())
                    tmp_path = Path(tmp.name)

                with st.spinner("Sampling frames, detecting faces, and classifying..."):
                    result = predictor.predict(tmp_path)

                if result["num_frames_with_face"] == 0:
                    st.warning(result["error"])
                else:
                    render_verdict(result["label"], result["confidence"])
                    st.caption(
                        f"Mean fake probability: {result['mean_fake_probability']:.2%} "
                        f"across {result['num_frames_with_face']} analyzed frames"
                    )
                    with st.expander("Per-frame breakdown"):
                        for i, fp in enumerate(result["frame_predictions"]):
                            st.write(f"Frame {i+1}: {fp['label']} ({fp['confidence']:.1%})")

    with tab_about:
        st.markdown(
            f"""
            **How it works**
            1. A face is detected and cropped (MTCNN).
            2. The crop is passed through a `{CFG.train.model_name}` classifier
               fine-tuned to distinguish real faces from GAN-generated/deepfake faces.
            3. For video, multiple frames are sampled and the average fake
               probability across frames determines the final verdict.

            **Limitations**
            - Trained on GAN-generated still-face images; performance on
              face-swap or lip-sync video deepfakes (different artifact type)
              may vary — see the README for planned FaceForensics++ extension.
            - Requires a clearly visible, front-facing face to run a prediction.
            """
        )


if __name__ == "__main__":
    main()
