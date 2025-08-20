import streamlit as st
import premium, risk_profile, dashboard, qa, Upload

# --- Page Config must be first ---
st.set_page_config(page_title="Insurance Underwriting System", layout="wide")

# --- CSS for styling (keep your design as is) ---
st.markdown("""
<style>
.st-emotion-cache-zuyloh {
    border: none; 
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    width: 100%;
    height: 100%;
    overflow: visible;
}
.st-emotion-cache-umot6g {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 400;
    padding: 8px 12px;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 4px 0;
    line-height: 1.6;
    font-size: inherit;
    font-family: inherit;
    color: inherit;
    width: 200px;
    cursor: pointer;
    background-color: rgb(43, 44, 54);
    border: 1px solid rgba(250, 250, 250, 0.2);
}
.st-emotion-cache-umot6g:hover {
    border-color: #4CAF50;
    color: #4CAF50;
}
.st-emotion-cache-umot6g:active {
    color: #fff;
    border-color: #4CAF50;
    background-color: #4CAF50;
}
.st-emotion-cache-umot6g:focus:not(:active) {
    border-color: #4CAF50;
    color: #4CAF50;
}
.st-emotion-cache-9ajs8n h4 {
    font-size: 14px;
    font-weight: 600;
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize page state ---
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://i.postimg.cc/W4ZNtNxP/usti-logo-1.png", use_container_width=True)
    st.markdown("Welcome To United Insurance Underwritter System")

    st.title("Navigation")

    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("Upload Files"):
        st.session_state.page = "Upload"
    if st.button("Risk Calculation"):
        st.session_state.page = "Risk"
    if st.button("Premium Calculation"):
        st.session_state.page = "Premium"
    if st.button("QA"):
        st.session_state.page = "QA"
    if st.button("Logout"):
        st.session_state.page = "Logout"

# --- Render selected page ---
if st.session_state.page == "Dashboard":
    dashboard.show()
elif st.session_state.page == "Upload":
    Upload.show()
elif st.session_state.page == "Risk":
    risk_profile.show()
elif st.session_state.page == "Premium":
    premium.show()
elif st.session_state.page == "QA":
    qa.show()
elif st.session_state.page == "Logout":
    st.success("You have been logged out successfully.")
