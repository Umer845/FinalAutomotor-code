import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sqlalchemy import create_engine, text
import datetime

# --- Database Configuration ---
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"
DB_TABLE_NAME1= "premium_results"

# --- Create SQLAlchemy engine ---
@st.cache_resource
def get_engine():
    """Establish and cache DB connection engine"""
    try:
        engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return create_engine(engine_string)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None


def show():
    """Streamlit App: Premium Prediction"""
    st.title("Motor Insurance Premium Prediction")

    # Connect DB
    engine = get_engine()
    if not engine:
        st.stop()

    # Load Model
    try:
        model = CatBoostRegressor()
        model.load_model("models/catboost_premium_model_2.pkl")
        feature_cols = joblib.load("models/model_features_2.pkl")
    except Exception as e:
        st.error(f"❌ Model not found or could not be loaded: {e}")
        model = None

    # --- Input Form ---
    st.subheader("Enter Vehicle Details")
    current_year = datetime.datetime.now().year

    with st.form(key="predict_form"):
        vehicle_make = st.text_input("Vehicle Make", value="Toyota")
        vehicle_model = st.text_input("Vehicle Model", value="Corolla")
        vehicle_make_year = st.number_input("Vehicle Make Year", min_value=1980, max_value=current_year, value=2020)
        sum_insured = st.number_input("Sum Insured", min_value=10000, value=500000)

        risk_profile = st.selectbox(
            "Select Risk Profile",
            ["Low", "Low to Moderate", "Moderate to High", "High"]
        )

        submit = st.form_submit_button("Predict Premium")

    if submit:
        make_normalized = vehicle_make.strip()
        model_normalized = vehicle_model.strip()
        vehicle_age = current_year - vehicle_make_year

        # --- Fetch Historical Rates ---
        try:
            with engine.connect() as conn:
                query = text(f"""
                    SELECT MIN(rate) as min_rate, 
                           MAX(rate) as max_rate, 
                           AVG(rate) as avg_rate
                    FROM {DB_TABLE_NAME}
                    WHERE LOWER(vehicle_make) = LOWER(:make)
                      AND LOWER(vehicle_model) = LOWER(:model);
                """)
                result = pd.read_sql(query, conn, params={"make": make_normalized, "model": model_normalized})
                historical_min_rate = result.iloc[0]["min_rate"]
                historical_max_rate = result.iloc[0]["max_rate"]
                historical_avg_rate = result.iloc[0]["avg_rate"]
        except Exception as e:
            st.error(f"❌ Error fetching historical data: {e}")
            historical_min_rate, historical_max_rate, historical_avg_rate = None, None, None

        # --- Prediction Logic ---
        if historical_avg_rate is not None:
            predicted_rate = historical_avg_rate
            predicted_premium = (predicted_rate / 100) * sum_insured
            prediction_source = "📊 Based on historical database rates"
        else:
            # ML Fallback
            if model:
                input_dict = {col: 0 for col in feature_cols}
                input_dict["VEHICLE MAKE"] = make_normalized
                input_dict["VEHICLE MODEL"] = model_normalized
                input_dict["VEHICLE MAKE YEAR"] = vehicle_make_year
                input_dict["SUM INSURED"] = sum_insured
                input_dict["vehicle_age"] = vehicle_age

                input_df = pd.DataFrame([input_dict])
                monthly_premium = model.predict(input_df)[0]
                predicted_premium = monthly_premium * 12
                predicted_rate = (predicted_premium / sum_insured * 100) if sum_insured else 0
                prediction_source = "🤖 Based on ML model"
            else:
                st.warning("⚠️ No historical data and model not loaded.")
                return

        # --- Vehicle Age Adjustment ---
        if vehicle_age <= 1:
            age_adjustment = 0.85
        elif 2 <= vehicle_age <= 5:
            age_adjustment = 1.00
        elif 6 <= vehicle_age <= 10:
            age_adjustment = 1.10
        else:
            age_adjustment = 1.25

        predicted_premium *= age_adjustment
        predicted_rate = (predicted_premium / sum_insured * 100) if sum_insured else 0

        # --- Risk Adjustment ---
        risk_factor_map = {
            "Low": 0.05,
            "Low to Moderate": 0.075,
            "Moderate to High": 0.10,
            "High": 0.15
        }
        risk_factor = risk_factor_map.get(risk_profile, 0)
        final_premium = predicted_premium * (1 + risk_factor)
        final_rate = (final_premium / sum_insured * 100) if sum_insured else 0

        # --- Display Results ---
        st.markdown("###  Prediction Results:")

        if historical_min_rate and historical_max_rate:
         st.markdown(
        f"""
        <div style="background-color:#0d5fafd9; padding:12px; border-radius:8px; margin-top:10px; margin-bottom:10px;">
            <span style="font-weight:bold; color:white;">
                Minimum Premium Rate: {historical_min_rate:.2f}% || Maximum Premium Rate: {historical_max_rate:.2f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
            f"""
            <div style="background-color:#1b6706e6; padding:12px; border-radius:8px; margin-top:10px; margin-bottom:10px;">
                <span style="font-weight:bold; color:white;">
                   Predicted Premium: {predicted_premium:,.2f} || Predicted Premium Rate: {predicted_rate:.2f}%
                </span>
            </div>
            """,
        unsafe_allow_html=True
        )

        st.markdown(
            f"""
                <div style="background-color:#735109; padding:12px; border-radius:8px; margin-top:10px; margin-bottom:10px;">
                   <span style="font-weight:bold; color:white;">
                       Final Premium with Risk {final_premium:,.2f} || Final Premium Rate with Risk {final_rate:.2f}%
                   </span>
                </div>

            """,
        unsafe_allow_html=True
        )

        # --- Store Data into Database ---
        try:
            with engine.begin() as conn:
                insert_query = text(f"""
                    INSERT INTO {DB_TABLE_NAME1}
                    (vehicle_make, vehicle_model, vehicle_make_year, sum_insured, vehicle_age, risk_profile,
                     historical_min_rate, historical_max_rate, historical_avg_rate,
                     predicted_premium, predicted_rate, final_premium, final_rate,
                     prediction_source, created_at)
                    VALUES (:make, :model, :year, :sum, :age, :risk,
                            :hist_min, :hist_max, :hist_avg,
                            :pred_premium, :pred_rate, :final_premium, :final_rate,
                            :prediction_source, :created_at)
                """)
                conn.execute(insert_query, {
                    "make": make_normalized,
                    "model": model_normalized,
                    "year": vehicle_make_year,
                    "sum": sum_insured,
                    "age": vehicle_age,
                    "risk": risk_profile,
                    "hist_min": float(historical_min_rate) if historical_min_rate else None,
                    "hist_max": float(historical_max_rate) if historical_max_rate else None,
                    "hist_avg": float(historical_avg_rate) if historical_avg_rate else None,
                    "pred_premium": float(predicted_premium),
                    "pred_rate": float(predicted_rate),
                    "final_premium": float(final_premium),
                    "final_rate": float(final_rate),
                    "prediction_source": prediction_source,
                    "created_at": datetime.datetime.now()
                })
            st.success("✅ Prediction saved to database successfully!")
        except Exception as e:
            st.error(f"❌ Failed to save data to DB: {e}")

if __name__ == "__main__":
    show()
