from flask import Flask, request, render_template
import numpy as np

from src.pipelines.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app= application

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict-data", methods=["POST", "GET"])
def predict_data():
    if request.method == "GET":
        return render_template("predict_page.html")
    else:
        CustomDataObj = CustomData(
            gender=request.form.get("gender"), 
            race_ethnicity=request.form.get("race_ethnicity"), 
            parental_level_of_education=request.form.get("parental_level_of_education"), 
            lunch=request.form.get("lunch"), 
            test_preparation_course=request.form.get("test_preparation_course"), 
            reading_score=int(request.form.get("reading_score")), 
            writing_score=int(request.form.get("writing_score"))
        )
        output_df = CustomDataObj.convert_to_dataframe()

        PredictionPipelineObj = PredictionPipeline()
        prediction = PredictionPipelineObj.predict(output_df)
        prediction = np.round(prediction[0], 1)

        return render_template("predict_page.html", results=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)