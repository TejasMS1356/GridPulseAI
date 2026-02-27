from flask import Flask, render_template, request, session, send_file, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model as tf_load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
import io
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

if not os.path.exists("static"):
    os.makedirs("static")

forecast_model = tf_load_model("load_forecast_model.h5")
scaler = joblib.load("scaler.pkl")

transformer_model = joblib.load("gridguard_transformer_model.pkl")

anomaly_model = joblib.load("xgb_anomaly_model.pkl")

SEQUENCE_LENGTH = 24
MULTIPLIER = 1000
anomaly_cache = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

@app.route("/")
def login_page():
    """Root – login page."""
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    """Simulate login – set session and redirect to home."""
    session['logged_in'] = True
    return redirect(url_for('home_page'))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page (GET) and simulate registration (POST)."""
    if request.method == "POST":
        session['logged_in'] = True
        return redirect(url_for('home_page'))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route("/home")
@login_required
def home_page():
    return render_template("home.html")

@app.route("/forecast", methods=["GET"])
@login_required
def forecast_page():
    """Serve the load forecasting page (upload form)."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    df = pd.read_csv(file)
    if "Load" not in df.columns:
        return "CSV must contain 'Load' column"

    load_values = df["Load"].values.reshape(-1, 1)
    scaled_data = scaler.transform(load_values)
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    y_pred = forecast_model.predict(X)

    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_actual_inv = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred_inv = y_pred_inv * MULTIPLIER
    y_actual_inv = y_actual_inv * MULTIPLIER

    mae = mean_absolute_error(y_actual_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_actual_inv, y_pred_inv))

    plt.figure(figsize=(10, 5))
    plt.plot(y_actual_inv, label="Actual Load (MW)")
    plt.plot(y_pred_inv, label="Predicted Load (MW)")
    plt.title("Actual vs Predicted Load (MW)")
    plt.ylabel("Load (MW)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/actual_vs_pred.png")
    plt.close()

    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    current_input = last_sequence.reshape(1, SEQUENCE_LENGTH, 1)
    future_predictions = []
    for _ in range(24):
        next_pred = forecast_model.predict(current_input)[0][0]
        future_predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_predictions = future_predictions * MULTIPLIER

    plt.figure(figsize=(10, 5))
    plt.plot(future_predictions)
    plt.title("Next 24 Hours Forecast (MW)")
    plt.ylabel("Load (MW)")
    plt.xlabel("Future Hours")
    plt.tight_layout()
    plt.savefig("static/future_forecast.png")
    plt.close()

    return render_template(
        "index.html",
        mae=str(round(mae, 2)) + " MW",
        rmse=str(round(rmse, 2)) + " MW",
        graph1="static/actual_vs_pred.png",
        graph2="static/future_forecast.png"
    )

@app.route("/gridguard", methods=["GET", "POST"])
@login_required
def gridguard():
    if request.method == "POST":
        time_hour = int(request.form["time_hour"])
        current_load = float(request.form["current_load"])
        temperature = float(request.form["temperature"])
        area = request.form["area"]
        consumers = int(request.form["consumers"])

        input_df = pd.DataFrame({
            'Time (hour)': [time_hour],
            'Current_Load_MW': [current_load],
            'Temperature_C': [temperature],
            'Area_Type': [area],
            'Consumers': [consumers]
        })

        input_encoded = pd.get_dummies(input_df, columns=['Area_Type'], drop_first=True)
        expected_columns = ['Time (hour)', 'Current_Load_MW', 'Temperature_C', 'Consumers',
                            'Area_Type_Industrial', 'Area_Type_Residential']
        input_processed = input_encoded.reindex(columns=expected_columns, fill_value=0)

        predicted_load = transformer_model.predict(input_processed)[0]
        transformer_capacity = 20
        utilization = (predicted_load / transformer_capacity) * 100

        if utilization < 80:
            risk = "NORMAL"
            risk_color = "green"
        elif utilization < 100:
            risk = "WARNING"
            risk_color = "orange"
        else:
            risk = "CRITICAL"
            risk_color = "red"

        np.random.seed(42)
        temp_data = np.random.uniform(15, 40, 200)
        load_data = 0.5 * temp_data + np.random.normal(5, 2, 200)

        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.scatter(temp_data, load_data, alpha=0.6)
        ax1.set_title("Load vs Temperature Relationship")
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Load (MW)")
        plt.tight_layout()
        plt.savefig("static/gridguard_plot1.png")
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.bar(["Utilization"], [utilization], color='blue')
        ax2.axhline(80, linestyle="--", color='orange', label='Warning (80%)')
        ax2.axhline(100, linestyle="--", color='red', label='Critical (100%)')
        ax2.set_ylim(0, 120)
        ax2.set_ylabel("Utilization (%)")
        ax2.set_title("Transformer Utilization Level")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("static/gridguard_plot2.png")
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.bar(["Current Load", "Predicted Load"], [current_load, predicted_load],
                color=['blue', 'orange'])
        ax3.set_title("Current vs Predicted Load (MW)")
        ax3.set_ylabel("Load (MW)")
        plt.tight_layout()
        plt.savefig("static/gridguard_plot3.png")
        plt.close()

        return render_template("gridguard.html",
                               time_hour=time_hour,
                               current_load=current_load,
                               temperature=temperature,
                               area=area,
                               consumers=consumers,
                               predicted_load=round(predicted_load, 2),
                               utilization=round(utilization, 1),
                               risk=risk,
                               risk_color=risk_color,
                               plot1="static/gridguard_plot1.png",
                               plot2="static/gridguard_plot2.png",
                               plot3="static/gridguard_plot3.png")
    else:
        return render_template("gridguard.html")

@app.route("/anomaly", methods=["GET", "POST"])
@login_required
def anomaly():
 
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        df = pd.read_csv(file)

        rename_map = {
            "weekday": "day_of_week",
            "rolling_mean": "rolling_mean_24",
            "rolling_std": "rolling_std_24"
        }
        df = df.rename(columns=rename_map)

        model_features = list(anomaly_model.feature_names_in_)
        missing = [col for col in model_features if col not in df.columns]
        if missing:
            return f"Missing columns: {missing}. Please ensure your CSV contains: {model_features}"

        X = df[model_features].copy()
        if hasattr(anomaly_model, "predict_proba"):
            probs = anomaly_model.predict_proba(X)[:, 1]
            df["Risk_Score"] = probs * 100
            df["Predicted_Anomaly"] = (probs > 0.5).astype(int)
        else:
            predictions = anomaly_model.predict(X)
            df["Predicted_Anomaly"] = predictions
            df["Risk_Score"] = predictions * 100

        cust_summary = df.groupby("Customer_ID").agg(
            Total_Records=("Predicted_Anomaly", "count"),
            Anomalies=("Predicted_Anomaly", "sum"),
            Avg_Risk_Score=("Risk_Score", "mean")
        ).reset_index()
        cust_summary["Anomalies"] = cust_summary["Anomalies"].astype(int)
        cust_summary["Avg_Risk_Score"] = cust_summary["Avg_Risk_Score"].round(2)
        abnormal_customers = cust_summary[cust_summary["Anomalies"] >= 3]

        uid = str(uuid.uuid4())
        anomaly_cache[uid] = {
            "df": df,
            "cust_summary": cust_summary,
            "abnormal_customers": abnormal_customers,
            "model_features": model_features
        }
        session["anomaly_uid"] = uid
        return redirect(url_for('anomaly'))

    uid = session.get("anomaly_uid")
    if uid and uid in anomaly_cache:
        data = anomaly_cache[uid]
        df = data["df"]
        cust_summary = data["cust_summary"]
        abnormal_customers = data["abnormal_customers"]

        selected_customer = request.args.get("customer")
        plot_path = None
        if selected_customer and selected_customer in df["Customer_ID"].values:
            cust_data = df[df["Customer_ID"] == selected_customer].copy()
            plt.figure(figsize=(10, 4))
            if "Datetime" in cust_data.columns:
                cust_data["Datetime"] = pd.to_datetime(cust_data["Datetime"])
                x_axis = cust_data["Datetime"]
            else:
                x_axis = cust_data.index
            plt.plot(x_axis, cust_data["Consumption_kWh"], label="Consumption", alpha=0.7)
            anomalies = cust_data[cust_data["Predicted_Anomaly"] == 1]
            if not anomalies.empty:
                plt.scatter(anomalies["Datetime"] if "Datetime" in anomalies.columns else anomalies.index,
                            anomalies["Consumption_kWh"], color="red", label="Anomaly", s=40)
            plt.title(f"Consumption Pattern for Customer {selected_customer}")
            plt.xlabel("Time")
            plt.ylabel("kWh")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_filename = f"anomaly_plot_{uid}_{selected_customer}.png"
            plt.savefig(f"static/{plot_filename}")
            plt.close()
            plot_path = f"static/{plot_filename}"

        total_records = len(df)
        total_anomalies = int(df["Predicted_Anomaly"].sum())
        high_risk_count = abnormal_customers.shape[0]
        avg_risk = df["Risk_Score"].mean()

        return render_template(
            "anomaly.html",
            show_results=True,
            uid=uid,
            total_records=total_records,
            total_anomalies=total_anomalies,
            high_risk_count=high_risk_count,
            avg_risk=round(avg_risk, 2),
            abnormal_customers=abnormal_customers.to_dict(orient="records"),
            customer_ids=df["Customer_ID"].unique().tolist(),
            selected_customer=selected_customer,
            plot_path=plot_path
        )
    else:
        return render_template("anomaly.html", show_results=False)

@app.route("/anomaly/download_summary")
@login_required
def download_summary():
    uid = session.get("anomaly_uid")
    if not uid or uid not in anomaly_cache:
        return "No data found", 404
    cust_summary = anomaly_cache[uid]["cust_summary"]
    csv_data = cust_summary.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="customer_anomaly_summary.csv"
    )

@app.route("/anomaly/download_detailed")
@login_required
def download_detailed():
    uid = session.get("anomaly_uid")
    if not uid or uid not in anomaly_cache:
        return "No data found", 404
    df = anomaly_cache[uid]["df"]
    csv_data = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="detailed_anomaly_results.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)