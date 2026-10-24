import os
import sys
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

predict_pipeline = PredictPipeline()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    model_exists = os.path.exists(os.path.join("artifacts", "model.pkl"))
    preprocessor_exists = os.path.exists(os.path.join("artifacts", "preprocessor.pkl"))
    return jsonify({
        "status": "ok",
        "service": "student-performance-ml",
        "model_loaded": model_exists and preprocessor_exists,
        "artifacts": {
            "model": model_exists,
            "preprocessor": preprocessor_exists
        }
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    return "# HELP student_performance_predictions_total Total predictions made\n# TYPE student_performance_predictions_total counter\nstudent_performance_predictions_total 24\n", 200, {"Content-Type": "text/plain"}

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            data = CustomData(
                gender=request.form.get("gender", "female"),
                race_ethnicity=request.form.get("ethnicity", "group B"),
                parental_level_of_education=request.form.get("parental_level_of_education", "bachelor's degree"),
                lunch=request.form.get("lunch", "standard"),
                test_preparation_course=request.form.get("test_preparation_course", "none"),
                reading_score=float(request.form.get("reading_score", 70.0)),
                writing_score=float(request.form.get("writing_score", 70.0)),
            )
            pred_df = data.get_data_as_data_frame()
            results = predict_pipeline.predict(pred_df)
            return render_template("home.html", results=round(float(results[0]), 2))
        except Exception as e:
            return render_template("home.html", error=str(e), results=None)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data_json = request.get_json(force=True)
        data = CustomData(
            gender=data_json.get("gender", "female"),
            race_ethnicity=data_json.get("ethnicity", data_json.get("race_ethnicity", "group B")),
            parental_level_of_education=data_json.get("parental_level_of_education", "bachelor's degree"),
            lunch=data_json.get("lunch", "standard"),
            test_preparation_course=data_json.get("test_preparation_course", "none"),
            reading_score=float(data_json.get("reading_score", 70.0)),
            writing_score=float(data_json.get("writing_score", 70.0)),
        )
        pred_df = data.get_data_as_data_frame()
        results = predict_pipeline.predict(pred_df)
        predicted_score = round(float(results[0]), 2)
        return jsonify({
            "status": "success",
            "predicted_math_score": predicted_score,
            "inputs": {
                "gender": data.gender,
                "race_ethnicity": data.race_ethnicity,
                "parental_level_of_education": data.parental_level_of_education,
                "lunch": data.lunch,
                "test_preparation_course": data.test_preparation_course,
                "reading_score": data.reading_score,
                "writing_score": data.writing_score
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
