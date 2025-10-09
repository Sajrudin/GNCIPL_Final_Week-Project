import joblib

PIPELINE_PATH = "Model/preprocess_pipeline.pkl"

try:
    pipeline = joblib.load(PIPELINE_PATH)
    print("✅ Pipeline loaded successfully!")
except EOFError:
    print("❌ EOFError: The file may be truncated or incomplete.")
except Exception as e:
    print(f"❌ Error loading pickle: {e}")
