import streamlit as st
import psycopg2
from train_model_2 import calculate_risk_score  # Reuse function
from datetime import datetime

# PostgreSQL connection details
DB_CONFIG = {
    "dbname": "AutoMotor_Insurance",
    "user": "postgres",
    "password": "United2025",
    "host": "localhost",
    "port": "5432"
}

def insert_risk_result(vehicle_use, vehicle_make_year, sum_insured, driver_age, vehicle_age, risk_score, risk_label):
    """Insert risk profile result into PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        insert_query = """
        INSERT INTO risk_profile_results
        (vehicle_use, vehicle_make_year, sum_insured, driver_age, vehicle_age, risk_score, risk_label)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
            vehicle_use,
            vehicle_make_year,
            float(sum_insured),
            int(driver_age),
            int(vehicle_age),
            float(risk_score),
            risk_label
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)

def show():
    st.title("Motor Insurance Risk Profile")

    with st.form(key="risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            vehicle_use = st.selectbox("Vehicle Use", ["personal", "commercial", "other"])
            vehicle_make_year = st.number_input("Vehicle Make Year", min_value=1980, max_value=2025, value=2020)

        with col2:
            sum_insured = st.number_input("Sum Insured", min_value=10000, value=500000)
            driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=30)

        submit = st.form_submit_button("Calculate Risk")

    if submit:
        current_year = datetime.now().year
        vehicle_age = current_year - vehicle_make_year
        risk_score, risk_label = calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age)

        # Save result to DB
        success, error = insert_risk_result(vehicle_use, vehicle_make_year, sum_insured, driver_age, vehicle_age, risk_score, risk_label)

        if success:
            st.success("✅ Risk profile saved to database.")
        else:
            st.error(f"❌ Database insert error: {error}")

        # Map labels to colors
        label_colors = {
            "Low": "#4CAF50",             # Green
            "Low to Moderate": "#9C27B0", # Purple
            "Medium to High": "#FF9800",  # Orange
            "High": "#F44336"             # Red
        }

        # Risk Score box
        st.markdown(
            f"""
            <div style="background-color:#2196F3; padding:15px; border-radius:8px; margin-bottom:10px;">
                <h4 style="color:white; margin:0;">Risk Score: {risk_score:.2f}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk Label box
        bg_color = label_colors.get(risk_label, "#555")
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:15px; border-radius:8px;">
                <h4 style="color:white; margin:0;">Risk Label: {risk_label}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    show()
