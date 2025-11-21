# Spam Detector Project

This project implements a **machine learning-based spam detector** using a scikit-learn pipeline with FastAPI for deployment.

```bash
1. Create & activate a virtual environment

Move spam detector elsewhere as you can't cd into 'Olivia Sabb' due to spaces, then open it in an IDE

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

Windows

python -m venv .venv
.venv\Scripts\activate

## 2. Install dependencies

pip install -r requirements.txt

(Make sure you select kernel)

3. Run Jupyter notebooks (prepare and train model)

jupyter notebook notebooks/01_eda.ipynb

    Explore the dataset and understand spam vs ham distribution.

Modeling

jupyter notebook notebooks/02_modeling.ipynb

    Train the model and export pipeline:

    notebooks/experiments/best_lr_model.joblib

API demo (optional)

    jupyter notebook notebooks/03_api_demo.ipynb

        Test predictions inside Jupyter if needed.

    Important: Make sure best_lr_model.joblib exists before running the API.

4. Run the FastAPI server

uvicorn src.serve.api:app --reload --port 8001

You should see:

Uvicorn running on http://127.0.0.1:8001

5. Test the API
Using CURL

curl -X POST "http://127.0.0.1:8001/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Congratulations! You won a prize!"}'

or, you can click on try it out and replace 

{
  "text": "string"
}

with something like

{
  "text": "Congratulations!!!!! You won a FREE Jeep Wrangler! Click the link below to accept"
}

Expected response:

{
  "spam_or_not": "spam",
  "probability_spam": 0.97
}

Using Swagger docs

Open in your browser: http://127.0.0.1:8001/docs

    Try the /predict endpoint

    Enter email text

    Click Execute to get prediction


6. Notes / Troubleshooting

    If you see errors like Expected 2D array, got 1D array, make sure the pipeline is loaded correctly and you’re passing a single string inside a list: [email_text].

    Ensure the model path is correct: notebooks/experiments/best_lr_model.joblib.

    Restart the server if port 8001 is already in use:
