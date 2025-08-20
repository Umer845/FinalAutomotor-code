import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor
import datetime
from sqlalchemy import create_engine, text
from rag_engine import answer_question  # ✅ your RAG engine (PDF vector DB)


st.markdown("""
<style>
ol{
font-size:14px;
font-weight:400;
}
</style>
""", unsafe_allow_html=True
)

# ----------------- Database Configuration -----------------
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"


# ----------------- Cache DB engine -----------------
@st.cache_resource
def get_engine():
    try:
        engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return create_engine(engine_string)
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None


# ----------------- Cache Model + Features -----------------
@st.cache_resource
def load_model():
    try:
        model = CatBoostRegressor()
        model.load_model("models/catboost_premium_model_2.pkl")
        feature_cols = joblib.load("models/model_features_2.pkl")
        return model, feature_cols
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None


# ----------------- Premium Calculation -----------------
def predict_premium(vehicle_make, vehicle_model, vehicle_make_year, sum_insured, risk_profile):
    engine = get_engine()
    model, feature_cols = load_model()
    if not engine or not model:
        raise Exception("DB engine or model not loaded")

    current_year = datetime.datetime.now().year
    vehicle_age = current_year - vehicle_make_year
    make_normalized = vehicle_make.strip()
    model_normalized = vehicle_model.strip()

    # --- Historical Rate Lookup ---
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT AVG(rate) as avg_rate
                FROM {DB_TABLE_NAME}
                WHERE LOWER(vehicle_make) = LOWER(:make)
                  AND LOWER(vehicle_model) = LOWER(:model);
            """)
            result = pd.read_sql(query, conn, params={"make": make_normalized, "model": model_normalized})
            historical_avg_rate = result.iloc[0]["avg_rate"]
    except Exception:
        historical_avg_rate = None

    # --- Prediction Logic ---
    if historical_avg_rate is not None and pd.notnull(historical_avg_rate):
        predicted_premium = (historical_avg_rate / 100) * sum_insured
    else:
        input_dict = {col: 0 for col in feature_cols}
        input_dict["VEHICLE MAKE"] = make_normalized
        input_dict["VEHICLE MODEL"] = model_normalized
        input_dict["VEHICLE MAKE YEAR"] = vehicle_make_year
        input_dict["SUM INSURED"] = sum_insured
        input_dict["vehicle_age"] = vehicle_age
        input_df = pd.DataFrame([input_dict])
        monthly_premium = model.predict(input_df)[0]
        predicted_premium = monthly_premium * 12

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

    # --- Risk Adjustment ---
    risk_factor_map = {"Low": 0.05, "Low to Moderate": 0.075, "Moderate to High": 0.10, "High": 0.15}
    risk_factor = risk_factor_map.get(risk_profile, 0)
    final_premium = predicted_premium * (1 + risk_factor)

    # --- Final Rate Calculation ---
    final_rate = (final_premium / sum_insured * 100) if sum_insured else 0

    return final_premium, final_rate


# ----------------- Streamlit UI -----------------
def show():
    # --- Initialize state for answer ---
    if "motor_insurance" not in st.session_state:
        st.session_state.motor_insurance = None

    st.title("Motor Insurance Questions")
    st.write("Do you need motor insurance?")

    # --- Smaller buttons using narrow columns ---
    col1, col2, col3, col4, col5 = st.columns([1,1,2,2,2])  # middle columns smaller

    with col1:
        if st.button("✅ Yes"):
            st.session_state.motor_insurance = "Yes"

    with col2:
        if st.button("❌ No"):
            st.session_state.motor_insurance = "No"

    # --- Show selection ---
    if st.session_state.motor_insurance:
        st.success(f"You selected: {st.session_state.motor_insurance}")

        # ---------------- If Yes -> Premium Calculation ----------------
        if st.session_state.motor_insurance == "Yes":
            st.subheader("🔑 Enter Vehicle Details")

            vehicle_make = st.text_input("Vehicle Make")
            if not vehicle_make: 
                return

            vehicle_model = st.text_input("Vehicle Model")
            if not vehicle_model: 
                return

            vehicle_year = st.number_input(
                "Vehicle Make Year", 
                min_value=1980, 
                max_value=datetime.datetime.now().year, 
                step=1
            )
            if not vehicle_year: 
                return

            sum_insured = st.number_input("Sum Insured", min_value=10000, step=1000)
            if not sum_insured: 
                return

            risk_profile = st.selectbox(
                "Risk Profile", 
                ["Low", "Low to Moderate", "Moderate to High", "High"]
            )

            if st.button("💡 Calculate Premium"):
                try:
                    final_premium, final_rate = predict_premium(
                        vehicle_make, vehicle_model, vehicle_year, sum_insured, risk_profile
                    )
                    st.success(f"💰 Final Premium: {final_premium:,.2f}")
                    st.info(f"📊 Final Premium Rate: {final_rate:.2f}%")
                except Exception as e:
                    st.error(f"⚠️ Error calculating premium: {e}")

        # ---------------- If No -> QA Section ----------------
        elif st.session_state.motor_insurance == "No":
            st.subheader("🔍 Ask a Question")
            query = st.text_input("Enter your question")

            if st.button("Ask"):
                if query.strip():
                    with st.spinner("🤔 Fetching answer..."):
                        try:
                            answer, sources = answer_question(query)
                        except Exception as e:
                            st.error(f"❌ Error fetching answer: {e}")
                            return

                    if not answer.strip():
                        st.warning("⚠️ No relevant answer found in the uploaded file.")
                    else:
                        st.markdown(
                            f"""
                            <div style="background-color:#19875440; padding:20px; border-radius:8px;">
                              <span style="font-size:16px; font-weight:600; color:white;">Answer with Explanation:</span> 
                              <p style="color:white; margin-top:5px; font-size:14px; font-weight:400">{answer}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("⚠️ Please enter a question first.")