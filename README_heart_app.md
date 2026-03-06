# Heart Disease ML – Interactive Report (Streamlit)

## How to run
1. Put `heart.csv` and (optionally) your saved model file (e.g., `log,model_joblib_heart`) in the same folder.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run heart_app_streamlit.py
   ```

The app will:
- Load the dataset
- Show dataset overview + EDA
- Train/evaluate models (or load your saved one if found)
- Show feature importance
- Provide a prediction playground
