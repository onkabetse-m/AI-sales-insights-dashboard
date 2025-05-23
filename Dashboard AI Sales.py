import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ✅ Load all models
reg_model = joblib.load("demo_regression_model.pkl")
clf_model = joblib.load("interaction_classifier.pkl")  # <--- This was missing
binary_model = joblib.load("binary_interaction_classifier.pkl")



st.set_page_config(page_title="AI Sales Insights Dashboard", layout="wide")
 
# Load and cache data
st.title("📊 AI Sales Insights Dashboard")
def load_data():
    df= pd.read_csv("AI_Sales_Insights_.csv")

try:
    df = pd.read_csv("AI_Sales_Insights_.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%H:%M:%S", errors='coerce')
    st.session_state["df"] = df
except FileNotFoundError:
    st.error(f"🚫 Could not find the file '{"AI_Sales_Insights_.csv"}'. Please make sure it's placed in the app folder.")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Visitor & Demo Insights",
    "Job & AI Interaction Insights",
    "Sales & Geography",
    "Session & Campaign Insights",
    "AI Predictions",
    "AI Binary Prediction",
    "Sales Target Analysis"  # ✅ Add this
])


# Sidebar actor and team filters
actor = st.sidebar.selectbox("View As", ["Executive", "Sales Team", "Marketing"])
df = st.session_state["df"]
 
# Sales Team filter
all_teams = df["Sales Team Name"].dropna().unique()
selected_teams = st.sidebar.multiselect("View by Sales Team", all_teams)
 
# Apply actor-specific filter
# Apply actor-specific filter
if actor == "Sales Team":
    if selected_teams:
        df = df[df["Sales Team Name"].isin(selected_teams)]  # filter selected teams
    # else: show all sales teams (no filter)
elif actor == "Marketing":
    df = df[df["Campaign Source"].isin(["Twitter", "Google Ads"])]
    if selected_teams:
        df = df[df["Sales Team Name"].isin(selected_teams)]
elif selected_teams:
    df = df[df["Sales Team Name"].isin(selected_teams)]
 
# Page Logic
if page == "Home":
    st.subheader(f"Welcome, {actor}!")
    st.write("Use the sidebar to navigate insights tailored to your role.")
    st.dataframe(df.head())
 
elif page == "Visitor & Demo Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("👥 Daily Unique Visitors")
        daily_visitors = df.groupby("Date")["IP Address"].nunique().reset_index()
        fig1 = px.bar(daily_visitors, x="Date", y="IP Address", labels={"IP Address": "Unique Visitors"})
        st.plotly_chart(fig1, use_container_width=True)
 
    with col2:
        st.subheader("🎯 Demo Request Conversion Rate")
        demo_requests = df[df["Resource Accessed"] == "/scheduledemo.php"]
        demo_conversions = demo_requests[demo_requests["Sale Made"] == "Yes"]
        demo_rate = (len(demo_conversions) / len(demo_requests)) * 100 if len(demo_requests) > 0 else 0
        st.metric("Demo Conversion Rate", f"{demo_rate:.2f}%")
 
elif page == "Job & AI Interaction Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("📈 Job Placement Volume")
        job_requests = df[df["Resource Accessed"] == "/jobportal.php"]
        job_volume = job_requests.groupby("Date").size().reset_index(name="Job Requests")
        fig2 = px.bar(job_volume, x="Date", y="Job Requests")
        st.plotly_chart(fig2, use_container_width=True)
 
    with col2:
        st.subheader("🤖 AI Assistant Interactions")
        ai_data = df[df["Resource Accessed"] == "/aiassistant.php"]
        ai_volume = ai_data.groupby("Date").size().reset_index(name="Interactions")
        fig3 = px.line(ai_volume, x="Date", y="Interactions")
        st.plotly_chart(fig3, use_container_width=True)
 
elif page == "Sales & Geography":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("🌍 Sales Distribution by Country")
        country_sales = df[df["Sale Made"] == "Yes"].groupby("Country")["Sale Price"].sum().reset_index()
        fig4 = px.pie(country_sales, values="Sale Price", names="Country", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)
 
    with col2:
        st.subheader("💼 Sales by Team")
        team_sales = df[df["Sale Made"] == "Yes"].groupby("Sales Team Name")["Sale Price"].sum().reset_index()
        fig5 = px.bar(team_sales, x="Sales Team Name", y="Sale Price")
        st.plotly_chart(fig5, use_container_width=True)
 
elif page == "Session & Campaign Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("⏱ Session Metrics")
        if "Session Duration" in df.columns:
            avg_time = df["Session Duration"].mean()
            st.metric("Avg. Session Time (sec)", f"{avg_time:.2f}")
        else:
            st.warning("Session Duration column is missing.")
 
        if "Session ID" in df.columns:
            pages_per_visit = df.groupby("Session ID")["Resource Accessed"].count().mean()
            st.metric("Pages per Visit", f"{pages_per_visit:.2f}")
        else:
            st.warning("Session ID column is missing.")
 
    with col2:
        st.subheader("📢 Campaign Conversion Rate")
        if "Sale Made" in df.columns:
            campaign_conv = df.groupby("Campaign Source")["Sale Made"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            campaign_conv.columns = ["Campaign Source", "Conversion Rate (%)"]
            fig6 = px.bar(campaign_conv, x="Campaign Source", y="Conversion Rate (%)")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("Sale Made column is missing or misformatted.")
 
elif page == "AI Predictions":
    st.subheader("🧠 AI Model Predictions")

    col1, col2 = st.columns(2)

    # 👉 REGRESSION: Predict demo requests
    with col1:
        st.markdown("### 📈 Predict Demo Requests")
        with st.form("regression_form"):
            day = st.number_input("Day", 1, 31, 15)
            month = st.number_input("Month", 1, 12, 5)
            year = st.number_input("Year", 2023, 2025, 2025)
            dow = st.number_input("Day of Week (0=Mon)", 0, 6, 2)
            reg_submit = st.form_submit_button("Predict")

        if reg_submit:
            reg_input = [[day, month, year, dow]]
            demo_prediction = reg_model.predict(reg_input)
            st.success(f"📊 Predicted Demo Requests: {int(demo_prediction[0])}")

    # 👉 CLASSIFICATION: Predict Interaction Type
    with col2:
        st.markdown("### 🧠 Predict Interaction Type")
        with st.form("classification_form"):
            c_day = st.number_input("Day", 1, 31, 15, key="c1")
            c_month = st.number_input("Month", 1, 12, 5, key="c2")
            c_year = st.number_input("Year", 2023, 2025, 2025, key="c3")
            c_dow = st.number_input("Day of Week (0=Mon)", 0, 6, 2, key="c4")
            clf_submit = st.form_submit_button("Classify")

        if clf_submit:
            clf_input = [[c_day, c_month, c_year, c_dow]]
            pred = clf_model.predict(clf_input)
            st.success(f"🔍 Predicted Interaction Type: {pred[0]}")

elif page == "AI Binary Prediction":
    st.subheader("🧠 AI Binary Interaction Classifier")
 
    with st.form("binary_form"):
        col1, col2 = st.columns(2)
 
        with col1:
            b_day = st.number_input("Day", 1, 31, 15, key="b1")
            b_month = st.number_input("Month", 1, 12, 5, key="b2")
            b_year = st.number_input("Year", 2023, 2025, 2025, key="b3")
            b_dow = st.number_input("Day of Week (0=Mon)", 0, 6, 2, key="b4")
            sale_made = st.selectbox("Sale Made", ["Yes", "No"], key="b5")
            status_code = st.selectbox("Status Code", [200, 304, 404, 500], key="b6")
 
        with col2:
            product = st.selectbox("Product", df["Product"].unique(), key="b7")
            team = st.selectbox("Sales Team Name", df["Sales Team Name"].unique(), key="b8")
            campaign = st.selectbox("Campaign Source", df["Campaign Source"].unique(), key="b9")
            country = st.selectbox("Country", df["Country"].unique(), key="b10")
 
        binary_submit = st.form_submit_button("Predict Interaction Type")
 
    if binary_submit:
        from sklearn.preprocessing import LabelEncoder
 
        # Encode like training
        def encode(col, val):
            encoder = LabelEncoder()
            encoder.fit(df[col])
            return encoder.transform([val])[0]
 
        product_enc = encode("Product", product)
        team_enc = encode("Sales Team Name", team)
        campaign_enc = encode("Campaign Source", campaign)
        country_enc = encode("Country", country)
        sale_val = 1 if sale_made == "Yes" else 0
 
        features = [[
            b_day, b_month, b_year, b_dow, product_enc, team_enc,
            campaign_enc, country_enc, sale_val, status_code
        ]]
 
        result = binary_model.predict(features)[0]
 
        if result == 1:
            st.success("✅ Important Interaction: (Demo, Job Request, or AI Assistant)")
        else:
            st.info("ℹ️ Likely classified as 'Other'")
 
elif page == "Sales Target Analysis":
    st.subheader("🎯 Sales Target Achievement by Team")

    # 🎯 Adjustable Target
    base_target = 75000000
    target_pct = st.slider("Set Target as % of original sales", 50, 150, 100, step=5) / 100
    SALES_TARGET = base_target * target_pct

    # 📊 Calculate achievements
    team_sales = df[df["Sale Made"] == "Yes"].groupby("Sales Team Name")["Sale Price"].sum().reset_index()
    team_sales["Target"] = SALES_TARGET
    team_sales["Achievement %"] = (team_sales["Sale Price"] / SALES_TARGET) * 100

    # 🎨 Color Logic
    def get_color(value):
        if value >= 100:
            return "green"
        elif value >= 90:
            return "orange"
        else:
            return "red"

    team_sales["Achievement Category"] = team_sales["Achievement %"].apply(get_color)

    # 📊 Bar Chart
    fig = px.bar(
        team_sales,
        x="Sales Team Name",
        y="Sale Price",
        color="Achievement Category",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=f"Team Sales vs Target (${SALES_TARGET:,.0f})",
        labels={"Sale Price": "Actual Sales", "Sales Team Name": "Team"}
    )

    # Add target line
    fig.add_hline(
        y=SALES_TARGET,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Target (${int(SALES_TARGET):,})",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="white",
        opacity=0.7
    )
       # 📏 KPI Thermometer: Total Sales Progress
    import plotly.graph_objects as go

    total_sales = team_sales["Sale Price"].sum()
    achievement_total_pct = (total_sales / SALES_TARGET) * 100

    kpi_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=achievement_total_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Total Sales Target Achievement (%)"},
        delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 90], 'color': "red"},
                {'range': [90, 100], 'color': "orange"},
                {'range': [100, 150], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))

    st.plotly_chart(kpi_fig, use_container_width=True)


   


    # 🔄 Layout: Bar + Donut + Legend
    col1, col2 = st.columns([3, 2])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔑 Legend")
        st.markdown("<span style='color:green'>🟩 Achieved</span>", unsafe_allow_html=True)
        st.markdown("<span style='color:orange'>🟧 Almost Achieved</span>", unsafe_allow_html=True)
        st.markdown("<span style='color:red'>🟥 Not Achieved</span>", unsafe_allow_html=True)

   