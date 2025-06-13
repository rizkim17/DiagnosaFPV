import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = "data_dummy_fpv_detailed.csv"  # Pastikan file ini ada di direktori yang sama

df = pd.read_csv(file_path)

# Ambil fitur dan label
feature_cols = [col for col in df.columns if col != "Diagnosis FPV"]
X = df[feature_cols]
y = df["Diagnosis FPV"]

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Simpan label encoder
joblib.dump(le, "label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "fpv_model.pkl")
print("Model dan encoder berhasil disimpan!")
