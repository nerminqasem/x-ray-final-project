from utils import *
import os
import json
import time
import uuid
from flask import Flask, render_template, url_for, request, redirect, jsonify, send_from_directory 
from werkzeug.utils import secure_filename
import shutil

# --------------------------------------------------------------------
# Main variables
# --------------------------------------------------------------------
MODEL_LIST = ['densenet121.pth', 'mobilenetv3.pth']
DEVICE = 'cpu'
THRESHOLD = 0.5

# Create the Flask application
app = Flask(__name__, static_url_path='/static')

# Allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = ['.dicom', '.dcm', '.png', '.jpg', '.jpeg']
ABS_PATH_TO_STATIC = os.path.abspath("static")

# Ensure required folders exist
for folder in [
    "uploads",
    "database",
    "static/results",
    "static/proc_images_dir",
    "static/pred_images_dir",
    "static/sample_images",
    "static/assets",
]:
    os.makedirs(folder, exist_ok=True)

# --------------------------------------------------------------------
# Database initialization
# --------------------------------------------------------------------
try:
    from app_db import (
        init_db, save_prediction, get_history, clear_history,
        get_experiments, get_logs, log_action
    )
    init_db()
    DB_AVAILABLE = True
    print("[OK] Database initialized.")
except Exception as e:
    print(f"[WARN] Database not available: {e}")
    DB_AVAILABLE = False

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
RESULTS_DIR = os.path.join("static", "results")

def _load_json(name, default=None):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fake_inference(model_name):
    metrics = _load_json("metrics.json", {})
    models = metrics.get("models", {})
    m = models.get(model_name, {})
    acc = m.get("accuracy", 0.90)
    h = sum(ord(c) for c in model_name)
    pred = "PNEUMONIA" if h % 2 == 0 else "NORMAL"
    confidence = float(min(0.99, max(0.70, acc + (h % 7) / 100.0)))
    inference_time = float(m.get("inference_time_ms", 30.0))
    return pred, confidence, inference_time


# ====================================================================
# HOME / ABOUT / FAQ
# ====================================================================
@app.route('/')
def home_func():
    return render_template('index.html')


@app.route('/faq')
def faq_func():
    return render_template('faq.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(
        os.path.abspath('uploads'),
        filename
    )

# ====================================================================
# PREPROCESSING PAGE
# ====================================================================
@app.route('/preprocessing')
def preprocessing_page():
    data = _load_json("preprocessing_comparison.json", {"preprocessing_comparison": []})
    return render_template("preprocessing.html", data=data.get("preprocessing_comparison", []))


# ====================================================================
# MODEL SELECTION PAGE
# ====================================================================
@app.route('/model-selection')
def model_selection_page():
    metrics = _load_json("metrics.json", {})
    models = metrics.get("models", {})
    comparison = _load_json("model_comparison.json", {"model_comparison": []})
    return render_template(
        "model_selection.html",
        models=models,
        comparison=comparison.get("model_comparison", [])
    )


# ====================================================================
# PREDICTION PAGE
# ====================================================================
@app.route('/prediction')
def prediction_page():
    return render_template("prediction.html")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get("image")
    preprocessing = request.form.get("preprocessing", "CLAHE")
    model_name = request.form.get("model", "DenseNet121")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".dcm", ".dicom"]:
        return jsonify({"error": "Unsupported file format"}), 400

    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join("uploads", fname)
    file.save(path)

    pred, conf, inf_time = _fake_inference(model_name)

    if DB_AVAILABLE:
        try:
            save_prediction(
                image_path=f"/{path}",
                prediction=pred,
                confidence=conf,
                preprocessing=preprocessing,
                model=model_name,
                inference_time=inf_time
            )
            log_action("predict", f"{model_name} | {preprocessing} | {pred}")
        except Exception as e:
            print(f"[WARN] Failed to save prediction: {e}")

    return jsonify({
    "image_url": url_for('uploaded_file', filename=fname),
    "preprocessed_url": url_for('uploaded_file', filename=fname),
    "heatmap_url": url_for('uploaded_file', filename=fname),
    "prediction": pred,
    "confidence": conf,
    "preprocessing": preprocessing,
    "model": model_name,
    "inference_time": inf_time
})


# ====================================================================
# DASHBOARD
# ====================================================================
@app.route('/dashboard')
def dashboard_page():
    metrics = _load_json("metrics.json", {})
    return render_template(
        "dashboard.html",
        metrics=metrics,
        dataset=metrics.get("dataset_statistics", {}),
        models=metrics.get("models", {}),
        preprocessing=metrics.get("preprocessing_comparison", []),
    )


# ====================================================================
# DATASET ANALYTICS
# ====================================================================
@app.route('/dataset')
def dataset_page():
    stats = _load_json("dataset_statistics.json", {})
    if not stats:
        metrics = _load_json("metrics.json", {})
        stats = metrics.get("dataset_statistics", {})
    return render_template("dataset_analytics.html", stats=stats)


# ====================================================================
# HISTORY
# ====================================================================
@app.route('/history')
def history_page():
    records = get_history() if DB_AVAILABLE else []
    return render_template("history.html", records=records)


@app.route('/api/history')
def api_history():
    return jsonify(get_history() if DB_AVAILABLE else [])


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    if DB_AVAILABLE:
        clear_history()
        log_action("history_clear")
    return jsonify({"status": "ok"})


# ====================================================================
# ADMIN
# ====================================================================
@app.route('/admin')
def admin_page():
    return render_template(
        "admin.html",
        history=get_history(limit=50) if DB_AVAILABLE else [],
        experiments=get_experiments() if DB_AVAILABLE else [],
        logs=get_logs(limit=100) if DB_AVAILABLE else [],
    )


# ====================================================================
# PUBLIC JSON APIs
# ====================================================================
@app.route('/api/metrics')
def api_metrics():
    return jsonify(_load_json("metrics.json", {}))


@app.route('/api/models')
def api_models():
    metrics = _load_json("metrics.json", {})
    return jsonify(metrics.get("models", {}))


@app.route('/api/dataset')
def api_dataset():
    stats = _load_json("dataset_statistics.json", {})
    if not stats:
        metrics = _load_json("metrics.json", {})
        stats = metrics.get("dataset_statistics", {})
    return jsonify(stats)


@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    method = request.form.get("method", "CLAHE")
    if DB_AVAILABLE:
        log_action("preprocess", method)
    return jsonify({
        "method": method,
        "status": "ok",
        "original_url": "/static/results/sample_original.png",
        "processed_url": f"/static/results/sample_{method.lower().replace(' ', '_')}.png"
    })


# ====================================================================
# LEGACY ROUTES (kept for backward compatibility)
# ====================================================================
@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    if os.path.isdir('uploads'):
        shutil.rmtree('uploads')
    if os.path.isdir('static/proc_images_dir'):
        shutil.rmtree('static/proc_images_dir')
    if os.path.isdir('static/pred_images_dir'):
        shutil.rmtree('static/pred_images_dir')

    if not os.path.isdir('uploads'):
        os.mkdir('uploads')

    file_list = request.files.getlist('my_files')
    for item in file_list:
        fname = item.filename
        extension = os.path.splitext(item.filename)[1].lower()
        if extension in app.config['ALLOWED_EXTENSIONS']:
            fname = secure_filename(fname)
            item.save(f'uploads/{fname}')

    upfile_list = os.listdir('uploads')
    image_list = []
    for item in upfile_list:
        file_ext = item.split('.')[1].lower()
        if f'.{file_ext}' in app.config['ALLOWED_EXTENSIONS']:
            image_list.append(item)

    process_images(image_list)

    num_preds_dict, sorted_image_list, disease_types_dict, detailed_results = predict_on_all_images(
        MODEL_LIST, DEVICE, image_list, THRESHOLD
    )

    image_fin_str = ""
    for i, item in enumerate(sorted_image_list):
        ext = item.split('.')[1]
        if ext in ['dcm', 'dicom']:
            item = item.replace(ext, 'png')
        image_fin_str += f"""<img class="w3-round unblock" src="/static/pred_images_dir/{item}" height="500">"""

    start_str = "<ul>"
    fin_str = start_str
    for i, item in enumerate(sorted_image_list):
        pred_info_tuple = num_preds_dict[item]
        lungs_detected = pred_info_tuple[1]
        num_preds = pred_info_tuple[0]
        disease_type = disease_types_dict.get(item, "Unknown")

        if lungs_detected == 'yes':
            if num_preds == 0:
                num_str = 'No disease detected'
            elif num_preds == 1:
                num_str = f'1 opacity detected - {disease_type}'
            else:
                num_str = f'{num_preds} opacities detected - {disease_type}'
        else:
            num_str = 'Error: The image is not a chest X-ray'

        if i == 0:
            fin_str += f'<li class="row w3-text-black w3-border-right w3-border-black w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'
        else:
            fin_str += f'<li class="row w3-padding-bottom" onclick="ajaxGetFilename(this.innerHTML)"><a href="#">{num_str}<br>{item}</a></li>'

    html_str = fin_str + '</ul>'

    first_fname = sorted_image_list[0]
    ext = first_fname.split('.')[1]
    if ext in ['dcm', 'dicom']:
        first_fname = first_fname.replace(ext, 'png')

    first_file_details = detailed_results.get(first_fname, {})
    diagnosis_text = generate_diagnosis_text(first_fname, first_file_details)

    main_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{first_fname}" height="500" alt="Chest X-ray">"""

    result_text_html = f"""
    <div id="diagnosis-result" class="diagnosis-box">
        <h4>Diagnosis Result</h4>
        <div class="diagnosis-content">
            {diagnosis_text}
        </div>
    </div>
    """

    output_response = {
        "html_str": html_str,
        "main_image_str": main_image_str,
        "image_fin_str": image_fin_str,
        "diagnosis_text": result_text_html
    }

    delete_user_submitted_data()
    return jsonify(output_response)


def generate_diagnosis_text(filename, details):
    num_opacities = details.get('num_opacities', 0)
    disease_type = details.get('disease_type', 'Unknown')
    confidence_scores = details.get('confidence_scores', [])
    locations = details.get('locations', [])

    text = f"<p><strong>File:</strong> {filename}</p>"

    if num_opacities == 0:
        text += "<p><strong>Result:</strong> No opacity detected. The X-ray appears normal.</p>"
        text += "<p><em>Note: This result is not a definitive diagnosis. Please consult a doctor.</em></p>"
    else:
        text += f"<p><strong>Result:</strong> {num_opacities} opacity(ies) detected.</p>"
        text += f"<p><strong>Primary disease type:</strong> {disease_type}</p>"
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) * 100
            text += f"<p><strong>Average confidence score:</strong> {avg_confidence:.1f}%</p>"
        text += "<p><strong>Opacity details:</strong></p><ul>"
        for i in range(num_opacities):
            conf = confidence_scores[i] * 100 if i < len(confidence_scores) else 0
            loc = locations[i] if i < len(locations) else "Not specified"
            text += f"<li>Opacity {i+1}: Location {loc}, Confidence {conf:.1f}%</li>"
        text += "</ul>"
        text += "<p><em>⚠️ This is an automated analysis. Always consult a radiology specialist for a definitive diagnosis.</em></p>"
    return text


@app.route('/process_ajax', methods=['POST'])
def process_ajax():
    fname = request.form.get('file_name')
    fname = fname.split('<br>')[1].replace('</a>', '')
    ext = fname.split('.')[1]
    image_fname = fname.split('.')[0] + '.png' if ext in ['dcm', 'dicom'] else fname
    diagnosis_text = generate_simple_diagnosis(image_fname)
    info_in_html = f"""
    <div class="image-with-diagnosis">
        <img id="selected-image" onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{image_fname}" height="500" alt="Chest X-ray">
        <div class="diagnosis-box">
            <h4>Diagnosis Result</h4>
            <div class="diagnosis-content">
                {diagnosis_text}
            </div>
        </div>
    </div>
    """
    return jsonify({"output1": info_in_html})


def generate_simple_diagnosis(filename):
    return f"<p><strong>File:</strong> {filename}</p><p>Analysis completed. Please refer to the results list for details.</p>"


@app.route('/sample_images', methods=['POST'])
def process_sample_ajax():
    fname = request.form.get('file_name')
    fname = fname.split('<br>')[1].replace('</a>', '')
    ext = fname.split('.')[1]
    image_fname = fname.split('.')[0] + '.png' if ext in ['dcm', 'dicom'] else fname

    if "tb0688.png" in image_fname:
        diagnosis_text = "<p><strong>Result:</strong> 1 opacity detected (Pneumonia)</p><p><strong>Confidence:</strong> 92%</p><p><strong>Location:</strong> Right lower lobe</p>"
    elif "114.png" in image_fname:
        diagnosis_text = "<p><strong>Result:</strong> 2 opacities detected (Consolidation)</p><p><strong>Confidence:</strong> 87%, 91%</p><p><strong>Location:</strong> Left upper lobe, right middle lobe</p>"
    else:
        diagnosis_text = "<p><strong>Result:</strong> No opacity detected (Normal)</p><p><strong>Confidence:</strong> 95%</p>"

    info_in_html = f"""
    <div class="image-with-diagnosis">
        <img id="selected-image" onclick="hide_show_bboxes(this.src, this.id)" class="w3-round unblock" src="/static/sample_images/{image_fname}" height="500" alt="Chest X-ray">
        <div class="diagnosis-box">
            <h4>Diagnosis Result</h4>
            <div class="diagnosis-content">
                {diagnosis_text}
                <p><em>⚠️ Automated analysis does not replace doctor's advice.</em></p>
            </div>
        </div>
    </div>
    """
    return jsonify({"output1": info_in_html})


@app.route('/process_click_info', methods=['POST'])
def process_click_info():
    fname = request.form.get('fname')
    image_fname = fname.split('/')[-1:][0]
    if 'no_bboxes_' in image_fname:
        output = show_all_bboxes(image_fname)
    else:
        output = hide_all_bboxes(image_fname)
    return jsonify(output)


@app.route('/process_sample_image_click', methods=['POST'])
def process_sample_image_click():
    fname = request.form.get('fname')
    id = request.form.get('id')
    image_fname = fname.split('/')[-1:][0]
    if 'noboxes_' in image_fname:
        new_fname = fname.replace('noboxes_', '')
        output = {'new_fname': new_fname, 'id': id}
    else:
        item_list = fname.split('/')
        image_fname = 'noboxes_' + image_fname
        last_index = len(item_list) - 1
        item_list[last_index] = image_fname
        new_fname = '/'.join(item_list)
        output = {'new_fname': new_fname, 'id': id}
    return jsonify(output)


@app.route('/test')
def test():
    return 'This is a test page...'


# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)