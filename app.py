import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

# ============================================
# Dashboard Title
# ============================================

st.title("Bank Personal Loan Approval Dashboard")

st.write(
    "This dashboard assists bank officers in analysing customer information "
    "and supporting personal loan approval decisions."
)

# ============================================
# Monitoring Dashboard
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# Prepare Dataset
X = df.drop('Personal Loan', axis=1)

# Remove non-model/dashboard columns
columns_to_drop = [
    'Education_Label',
    'CCAvg'          # drop because contains string values like '3/80'
]

for col in columns_to_drop:
    if col in X.columns:
        X = X.drop(col, axis=1)

# Keep only numeric columns as a safety check
X = X.select_dtypes(include=['int64', 'float64'])

y = df['Personal Loan']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Best Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)

# Monitoring Metrics

model_accuracy = accuracy_score(y_test, y_pred)
loan_recall = recall_score(y_test, y_pred)
missing_values = df.isnull().sum().sum()

# Dashboard Title
st.header("Monitoring Dashboard")

st.write(
    "Monitoring metrics are used to evaluate "
    "model performance and data quality after deployment."
)

### CSS Styling ###
st.markdown("""
<style>
.metric-card {
    background-color: #EAF4FF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #B3D9FF;
    text-align: center;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 18px;
    font-weight: bold;
    color: #003366;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #0066CC;
}
</style>
""", unsafe_allow_html=True)

# Display Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Model Accuracy
        </div>
        <div class="metric-value">
            {model_accuracy:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Loan Recall
        </div>
        <div class="metric-value">
            {loan_recall:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">
            Missing Values
        </div>
        <div class="metric-value">
            {missing_values}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Convert Education Codes
# ============================================

education_mapping = {
    1: "Undergraduate",
    2: "Graduate",
    3: "Advanced / Professional"
}

df["Education_Label"] = df["Education"].map(education_mapping)

# ============================================
# Predictive Input
# ============================================

st.header("Loan Approval Prediction")

st.write("Enter customer information below:")

age = st.number_input("Age", 18, 100, 35)

income = st.number_input("Annual Income (thousand)", 0, 300, 80)

experience = st.number_input("Years of Experience", 0, 50, 10)

ccavg = st.number_input("Average Credit Card Spending", 0.0, 10.0, 2.0)

mortgage = st.number_input("Mortgage", 0, 700, 0)

education_input = st.selectbox(
    "Education Level",
    list(education_mapping.values())
)

# ============================================
# Temporary Prediction Logic
# Replace with your trained model later
# ============================================

if st.button("Predict Loan Approval"):

    if income >= 100 and ccavg >= 3:
        st.success("Prediction: Loan Likely to be Approved")
    else:
        st.error("Prediction: Loan Likely to be Rejected")

# ============================================
# Filters
# ============================================
st.write(
    "DASHBOARD FILTER:"
)

income_range = st.slider(
    "Income Range",
    int(df["Income"].min()),
    int(df["Income"].max()),
    (50, 150)
)

education = st.selectbox(
    "Education Level",
    ["All"] + list(education_mapping.values())
)

# ============================================
# Apply Filters
# ============================================

filtered_df = df[
    df["Income"].between(income_range[0], income_range[1])
]

if education != "All":
    filtered_df = filtered_df[
        filtered_df["Education_Label"] == education
    ]

st.write(f"Number of Customers: **{len(filtered_df)}**")

# ============================================
# Visualization 1
# Income Distribution
# ============================================

st.subheader("1. Income Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(filtered_df["Income"], bins=20)

median_income = filtered_df["Income"].median()

ax.set_xlabel("Income")
ax.set_ylabel("Frequency")
ax.legend()

st.pyplot(fig)

# ============================================
# Visualization 2
# Income vs Personal Loan Approval
# ============================================

st.subheader("2. Income vs Personal Loan Approval")

fig, ax = plt.subplots(figsize=(8,5))

sns.stripplot(
    data=filtered_df,
    x="Income",
    y="Personal Loan",
    jitter=True,
    alpha=0.5,
    ax=ax
)

ax.set_yticks([0, 1])
ax.set_yticklabels(["Rejected", "Approved"])

ax.set_xlabel("Income")
ax.set_ylabel("Loan Approval Status")

st.pyplot(fig)

# ============================================
# Visualization 3
# Loan Approval Distribution
# ============================================

st.subheader("3. Loan Approval Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    data=filtered_df,
    x="Personal Loan",
    ax=ax
)

ax.set_xticklabels(["Rejected", "Approved"])

ax.set_xlabel("Loan Status")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)
