import pandas as pd

df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

def test_no_missing_values():
    assert df.isnull().sum().sum() == 0

def test_no_duplicates():
    assert df.duplicated().sum() == 0

def test_income_non_negative():
    assert (df["Income"] >= 0).all()

def test_age_range():
    assert df["Age"].between(18, 100).all()