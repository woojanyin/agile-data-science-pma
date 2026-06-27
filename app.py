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

st.title("🏦 Bank Personal Loan Approval Dashboard")

st.write(
    "This dashboard assists bank officers in analysing customer information "
    "and supporting personal loan approval decisions."
)

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
# Predictive Output
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

st.sidebar.header("Filter Customers")

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
