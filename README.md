# PneumoVision

An educational prototype for chest X-ray analysis. The repository combines a Flask web interface, image-handling utilities, SQLite-backed prediction history, model-comparison data, and experimental classification and object-detection code.

> **Important:** The main `/api/predict` endpoint currently returns a deterministic placeholder prediction; it does not run DenseNet121 or MobileNetV3. The displayed preprocessing and heatmap outputs are also placeholders. Metrics in the repository are predefined records and have not been reproduced by the supplied training pipeline. This software is for educational and research use only and must not be used for diagnosis or treatment decisions.

## Project Contents

| Path | Description |
|---|---|
| `x-ray/` | Main Flask application, templates, database layer, static files, and dependencies |
| `train_models/` | Classification model definitions, training/evaluation scripts, and recorded result files |
| `train_models/x-ray-model/` | Separate Faster R-CNN object-detection prototype |
| `x-ray-test/` | Standalone test application for the object-detection model |
| `static/`, `uploads/`, `database/` | Additional project assets and data folders at the repository root |
| `PROJECT_REPORT.md` | Academic project report |

## Main Application Features

- Flask pages for the home screen, prediction form, preprocessing comparison, model comparison, dashboard, dataset statistics, prediction history, and administration.
- SQLite functions for storing prediction records and application logs.
- Utilities for image handling, including DICOM reading and bounding-box drawing in the legacy object-detection workflow.
- JSON files containing dataset statistics and recorded model/preprocessing metrics.
- Dockerfile and Docker Compose configuration inside `x-ray/`.

Some visible interface options describe intended functionality rather than a complete implementation. In particular, the main prediction endpoint does not execute real model inference or apply the selected preprocessing method.

## Requirements

- Python 3.10 is recommended to match the supplied Dockerfile and pinned dependencies.
- Install the Python packages listed in `x-ray/requirements.txt`.
- DICOM handling and image processing depend on the packages listed there; installation availability may vary by operating system.

## Run Locally on Windows

Open PowerShell in the repository root and run:

```powershell
Set-Location .\x-ray
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The Flask application creates required upload and static folders and initializes its SQLite database on startup. Run it from the `x-ray/` directory because the application uses working-directory-relative paths. The development server is configured with debug mode enabled; do not expose it to untrusted networks or use it as a production deployment.

## Application Pages

- `/` — project home page
- `/prediction` — image upload and prediction interface
- `/preprocessing` — recorded preprocessing comparison
- `/model-selection` — recorded model comparison
- `/dashboard` — metrics charts and visualizations
- `/dataset` — dataset split summary
- `/history` — stored prediction records
- `/admin` — recent records and application logs

## Model and Evaluation Status

The classification configuration lists two classes, `NORMAL` and `PNEUMONIA`, and compares DenseNet121 with MobileNetV3. The result files report DenseNet121 accuracy of 93.42% and MobileNetV3 accuracy of 91.83% under CLAHE preprocessing. These are repository-reported values only:

- `train_models/train.py` writes predefined history values rather than training a model on image data.
- Model metrics are defined as constants in the model metric modules.
- The dataset loader returns configured summary values; the dataset path and full training data are not included in the repository structure.
- The main Flask prediction endpoint uses placeholder inference.
- The stated dataset total (5,863) differs from the sum of the listed train, validation, and test splits (5,856).

The separate Faster R-CNN code is a prototype and should not be considered a validated detector. Its inspected training scaffold does not load real annotations in its fallback dataset path.

## Data, Privacy, and Medical Use

Do not upload identifiable or sensitive patient information to this prototype. The current application does not implement administrator authentication or a complete clinical data-protection workflow. Any real-world use would require secure access controls, retention and deletion policies, privacy review, reproducible training, independent clinical validation, and qualified medical oversight.

## Further Reading

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the project description, methodology, reported results, limitations, and references.
