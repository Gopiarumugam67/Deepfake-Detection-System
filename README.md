# 🛡️ Deepfake Detection System

### AI-Powered Image & Video Authenticity Analysis

An end-to-end deep learning application that analyzes facial images and video frames to classify them as **Real or Fake** using an EfficientNet-B0 model. Built with Python, PyTorch, OpenCV, and Streamlit, this project combines model inference, face extraction, video analysis, evaluation, and automated testing in a practical AI application.

**Built with dedication by Gopinath A** — Computer Science Engineer specializing in Artificial Intelligence and Machine Learning.

[GitHub Repository](https://github.com/Gopiarumugam67/Deepfake-Detection-System) · [Developer Profile](https://github.com/Gopiarumugam67)

---

## 📌 Project Overview

The growing accessibility of synthetic media makes it increasingly important to analyze the authenticity of digital content. This project explores deep learning–based facial analysis to identify potential deepfakes in images and videos.

The application extracts facial regions, runs model inference, and produces a prediction with a confidence score. For videos, it samples frames, analyzes detected faces, and aggregates their fake probabilities into a video-level result.

The goal is to provide a practical demonstration of applied deep learning, computer vision, model evaluation, and application development.

## ✨ Key Features

- **Image Classification:** Analyze uploaded facial images and obtain a Real/Fake prediction.
- **Video Analysis:** Sample video frames and aggregate predictions into a single verdict.
- **Face Extraction:** Detect and extract facial regions before classification.
- **Confidence Scoring:** Display the model's prediction confidence.
- **Interactive Web Interface:** Use a Streamlit application for image and video analysis.
- **Model Evaluation:** Assess performance using classification metrics and ROC-AUC.
- **Error Handling:** Handle invalid or unreadable video files without producing a misleading verdict.
- **Automated Testing:** Validate dataset-building and inference logic using pytest.

## 🧠 Model & Performance

The system uses **EfficientNet-B0**, initialized with pretrained weights and fine-tuned for binary facial image classification.

### Test-set results

| Metric | Result |
|---|---:|
| Accuracy | 89.47% |
| Precision | 89.78% |
| Recall | 89.07% |
| F1-score | 89.42% |
| ROC-AUC | 96.27% |

**Training configuration**

- Architecture: EfficientNet-B0
- Input image size: 224 × 224
- Training duration: 20 epochs
- Training environment: Google Colab with NVIDIA Tesla T4 GPU
- Prepared dataset: 4,999 cropped facial images
- Dataset split: 3,499 training, 750 validation, and 750 test images

These results describe performance on the project's existing test split. They should not be interpreted as a guarantee of accuracy on unseen datasets or every deepfake generation technique.

## 🏗️ System Workflow

1. **Input:** Upload a facial image or video through the Streamlit interface.
2. **Validation:** Check the input and handle unsupported or unreadable video files.
3. **Preprocessing:** Detect and extract facial regions.
4. **Inference:** Run EfficientNet-B0 on the extracted facial images.
5. **Aggregation:** For videos, average the frame-level fake probabilities.
6. **Output:** Display the predicted label and confidence score.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, EfficientNet-B0 |
| Computer Vision | OpenCV, MTCNN |
| Web Application | Streamlit |
| Data Processing | NumPy |
| Testing | pytest |
| Development Tools | Git, GitHub, Google Colab |

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Git
- A compatible PyTorch environment
- The trained model checkpoint

### 1. Clone the repository

```bash
git clone https://github.com/Gopiarumugam67/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the model checkpoint

Ensure the trained checkpoint is available at:

```text
checkpoints/best_efficientnet_b0.pt
```

If the checkpoint is not present in your clone, obtain the trained checkpoint separately and place it at the expected path. Do not commit large datasets or model artifacts without checking repository size and licensing requirements.

### 5. Launch the application

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal, then choose the image or video analysis interface.

## 🧪 Testing

Run the automated test suite:

```powershell
python -m pytest -v
```

**Latest verified result:** 13 tests passed.

The current suite covers dataset-building behavior, image prediction outcomes, video prediction aggregation, and the no-face result path. Invalid-video handling has also been manually verified in the Streamlit application.

## 📁 Project Structure

```text
Deepfake-Detection-System/
├── app/
│   └── streamlit_app.py
├── checkpoints/
│   └── best_efficientnet_b0.pt
├── config/
├── src/
│   ├── inference/
│   │   ├── image_predictor.py
│   │   └── video_predictor.py
│   ├── preprocessing/
│   │   └── face_extractor.py
│   ├── models/
│   └── utils/
├── tests/
│   ├── test_dataset_builder.py
│   ├── test_image_predictor.py
│   └── test_video_predictor.py
├── requirements.txt
├── pytest.ini
└── README.md
```

*Note: The tree highlights key project components; other files and directories may exist in the repository.*

## ⚠️ Limitations

- Performance depends on the quality of the input, face detection, and the model's training data.
- The reported test metrics do not establish generalization to every unseen dataset.
- Performance on face-swap, lip-sync, and other deepfake techniques has not been independently established.
- A Real/Fake label is a model prediction, not definitive proof of authenticity or manipulation.
- This project is intended for educational and research demonstration purposes, not as a standalone forensic verification tool.

## 🔭 Future Improvements

- Evaluate generalization on diverse public deepfake benchmarks.
- Explore video-level temporal modeling instead of averaging frame predictions alone.
- Add explainability visualizations, such as Grad-CAM.
- Expand automated tests for malformed inputs and edge cases.
- Optimize inference latency and improve deployment readiness.

## 👨‍💻 About the Developer

**Gopinath A**  
Computer Science Engineer | Artificial Intelligence & Machine Learning

I developed this project to apply deep learning and computer vision to a practical media-authenticity problem. The work includes model training and evaluation, facial preprocessing, image and video inference, an interactive Streamlit interface, error handling, and automated testing.

My interests include AI/ML engineering, Python development, and building practical, end-to-end software systems.

- **GitHub:** [Gopiarumugam67](https://github.com/Gopiarumugam67)
- **Project:** [Deepfake Detection System](https://github.com/Gopiarumugam67/Deepfake-Detection-System)

*Designed, developed, and maintained by Gopinath A.*

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright © 2026 Gopinath A.

**Note:** Third-party libraries, pretrained models, and datasets may be subject to separate licenses and terms of use.

---

⭐ **If you find this project useful, consider starring the repository on GitHub.**

*Built by me (Gk) with Python, deep learning, and a commitment to continuous learning.*
