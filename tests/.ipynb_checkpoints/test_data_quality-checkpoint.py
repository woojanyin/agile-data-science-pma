import pandas as pd

df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

def test_no_missing_values():
    assert df.isnull().sum().sum() == 0