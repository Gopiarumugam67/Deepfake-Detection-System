# Deepfake Media Detection System

A deep learning application that classifies face images and videos as **Real** or **Fake** using EfficientNet-B0, PyTorch, MTCNN face detection, and Streamlit.

The system supports image-level predictions, video frame sampling, per-frame analysis, and an overall video verdict.

## Key Features

- Real/fake face classification using transfer learning.
- MTCNN-based face detection and cropping.
- Image and video upload through a Streamlit interface.
- Frame sampling and aggregated video-level predictions.
- Training with validation, early stopping, and checkpoint saving.
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- CPU inference support for local Windows deployment.

## Model and Dataset

**Model:** EfficientNet-B0  
**Framework:** PyTorch  
**Face detection:** MTCNN  
**Interface:** Streamlit

The project uses the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), containing real and GAN-generated face images.

For this training run, a prepared subset of **4,999 face images** was used, preserving the original dataset split:

| Split | Images |
|---|---:|
| Training | 3,499 |
| Validation | 750 |
| Testing | 750 |
| **Total** | **4,999** |

The face images were prepared before training, and CSV manifests were generated for the training, validation, and test splits.

## Model Performance

The model was trained for 20 epochs using a Tesla T4 GPU in Google Colab. The best checkpoint was selected using validation loss.

**Held-out test results:**

| Metric | Result |
|---|---:|
| Accuracy | 89.47% |
| Precision | 89.78% |
| Recall | 89.07% |
| F1-score | 89.42% |
| ROC-AUC | 96.27% |
| Test samples | 750 |

The best validation accuracy reached 90.53%.

Evaluation artifacts include a confusion matrix, ROC curve, classification report, and machine-readable metrics.

These results apply to the prepared test set. Performance on other datasets, real-world footage, face swaps, and lip-sync manipulations may differ.

## Project Structure

```text
deepfake-detection/
├── app/
│   └── streamlit_app.py
├── checkpoints/
│   └── best_efficientnet_b0.pt
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── logs/
├── reports/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── augmentation.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── inference/
│   │   ├── image_predictor.py
│   │   └── video_predictor.py
│   ├── models/
│   │   └── xception_model.py
│   ├── preprocessing/
│   │   ├── face_extractor.py
│   │   └── dataset_builder.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       └── logger.py
├── tests/
├── config.py
├── requirements.txt
└── README.md
```

## Run the Trained Application on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/Gopiarumugam67/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

If you already have the project locally, open its existing folder instead.

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If a dependency installation fails, resolve that package issue before continuing.

### 4. Add the trained checkpoint

Place the trained checkpoint at:

```text
checkpoints/best_efficientnet_b0.pt
```

The checkpoint is generated during training and is not necessarily included in the Git repository. Obtain it from the project's training artifacts.

### 5. Launch Streamlit

```powershell
python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

The application supports image uploads and video uploads. Video analysis samples frames, predicts each detected face, and aggregates the frame-level fake probabilities into a video-level verdict.

## Training and Evaluation

For model development, use a GPU-enabled environment such as Google Colab.

After preparing the dataset and creating the split manifests, run:

```bash
python -m src.training.trainer
python -m src.evaluation.evaluator
```

Training saves the best checkpoint under `checkpoints/` and the epoch history under `logs/`. Evaluation saves the metrics and visualizations under `reports/`.

**Important:** The preprocessing scripts can regenerate datasets and manifests. Preserve the existing split files if you need to reproduce the reported experiment.

## Running Tests

Install the development dependencies if the repository provides a `requirements-dev.txt` file, then run:

```bash
python -m pytest -v
```

Check the current test suite and its results before reporting a passing test count.

## Limitations

- The model was trained on a relatively small prepared subset of face images.
- Training data focused on real and GAN-generated still-face images; performance on face-swap and lip-sync video manipulations has not been established.
- Predictions depend on face visibility, image quality, and the input distribution.
- Confidence scores are model outputs, not calibrated guarantees of authenticity.
- Video predictions aggregate sampled frames and may miss manipulation artifacts between sampled frames.

## Future Improvements

- Evaluate on larger and more diverse datasets.
- Add FaceForensics++ or other video-manipulation datasets.
- Investigate Grad-CAM visualizations for model interpretability.
- Add a FastAPI inference endpoint.
- Improve robustness testing and document performance across manipulation types.

## Technologies

Python · PyTorch · EfficientNet-B0 · timm · OpenCV · MTCNN · Albumentations · scikit-learn · Streamlit · Google Colab
