"""
app_routes_extra.py
Extra Flask routes: preprocessing, model selection, prediction, dashboard,
dataset analytics, history, admin.
Import and register with: register_extra_routes(app)
"""
import os
import json
import time
import uuid
from flask import render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from app_db import (
    init_db, save_prediction, get_history, clear_history,
    save_experiment, get_experiments, log_action, get_logs,
)

RESULTS_DIR = os.path.join("static", "results")
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_json(name, default=None):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fake_inference(model_name):
    """
    Loads metadata from metrics JSON and returns a deterministic fake result.
    Replace with real inference later.
    """
    metrics = _load_json("metrics.json", {})
    models = metrics.get("models", {})
    m = models.get(model_name, {})
    acc = m.get("accuracy", 0.9)

    # Deterministic pseudo-random based on model name
    h = sum(ord(c) for c in model_name)
    pred = "PNEUMONIA" if h % 2 == 0 else "NORMAL"
    confidence = float(min(0.99, max(0.70, acc + (h % 7) / 100.0)))
    inference_time = float(m.get("inference_time_ms", 30.0))
    return pred, confidence, inference_time


def register_extra_routes(app):

    # ---------------- Preprocessing ----------------
    @app.route("/preprocessing")
    def preprocessing_page():
        data = _load_json("preprocessing_comparison.json", {"preprocessing_comparison": []})
        return render_template("preprocessing.html", data=data.get("preprocessing_comparison", []))

    @app.route("/api/preprocess", methods=["POST"])
    def api_preprocess():
        # For now just echoes back the method; real preprocessing runs via utils.process_images
        method = request.form.get("method", "CLAHE")
        log_action("preprocess", method)
        return jsonify({
            "method": method,
            "status": "ok",
            "original_url": "/static/results/sample_original.png",
            "processed_url": f"/static/results/sample_{method.lower().replace(' ', '_')}.png"
        })

    # ---------------- Model Selection ----------------
    @app.route("/model-selection")
    def model_selection_page():
        metrics = _load_json("metrics.json", {})
        models = metrics.get("models", {})
        comparison = _load_json("model_comparison.json", {"model_comparison": []})
        return render_template(
            "model_selection.html",
            models=models,
            comparison=comparison.get("model_comparison", [])
        )

    @app.route("/api/models")
    def api_models():
        metrics = _load_json("metrics.json", {})
        return jsonify(metrics.get("models", {}))

    # ---------------- Prediction ----------------
    @app.route("/prediction")
    def prediction_page():
        return render_template("prediction.html")

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        file = request.files.get("image")
        preprocessing = request.form.get("preprocessing", "CLAHE")
        model_name = request.form.get("model", "DenseNet121")

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            return jsonify({"error": "Unsupported file format"}), 400

        fname = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(UPLOADS_DIR, fname)
        file.save(path)

        start = time.time()
        pred, conf, inf_time = _fake_inference(model_name)
        total_time = (time.time() - start) * 1000

        save_prediction(
            image_path=f"/{path}",
            prediction=pred,
            confidence=conf,
            preprocessing=preprocessing,
            model=model_name,
            inference_time=inf_time
        )
        log_action("predict", f"{model_name} | {preprocessing} | {pred}")

        return jsonify({
            "image_url": f"/{path}",
            "prediction": pred,
            "confidence": conf,
            "preprocessing": preprocessing,
            "model": model_name,
            "inference_time": inf_time,
            "total_time": total_time
        })

    # ---------------- Dashboard ----------------
    @app.route("/dashboard")
    def dashboard_page():
        metrics = _load_json("metrics.json", {})
        return render_template(
            "dashboard.html",
            metrics=metrics,
            dataset=metrics.get("dataset_statistics", {}),
            models=metrics.get("models", {}),
            preprocessing=metrics.get("preprocessing_comparison", []),
        )

    @app.route("/api/metrics")
    def api_metrics():
        return jsonify(_load_json("metrics.json", {}))

    # ---------------- Dataset Analytics ----------------
    @app.route("/dataset")
    def dataset_page():
        stats = _load_json("dataset_statistics.json", {})
        if not stats:
            metrics = _load_json("metrics.json", {})
            stats = metrics.get("dataset_statistics", {})
        return render_template("dataset_analytics.html", stats=stats)

    @app.route("/api/dataset")
    def api_dataset():
        stats = _load_json("dataset_statistics.json", {})
        if not stats:
            metrics = _load_json("metrics.json", {})
            stats = metrics.get("dataset_statistics", {})
        return jsonify(stats)

    # ---------------- History ----------------
    @app.route("/history")
    def history_page():
        return render_template("history.html", records=get_history())

    @app.route("/api/history")
    def api_history():
        return jsonify(get_history())

    @app.route("/api/history/clear", methods=["POST"])
    def api_history_clear():
        clear_history()
        log_action("history_clear")
        return jsonify({"status": "ok"})

    # ---------------- Admin ----------------
    @app.route("/admin")
    def admin_page():
        return render_template(
            "admin.html",
            history=get_history(limit=50),
            experiments=get_experiments(),
            logs=get_logs(limit=100),
        )

    @app.route("/api/logs")
    def api_logs():
        return jsonify(get_logs())

    @app.route("/api/admin/experiments")
    def api_experiments():
        return jsonify(get_experiments())


# Make sure DB is initialized when this module is imported
init_db()