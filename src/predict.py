"""
Loan Approval Prediction - Inference / Prediction Script
Author: Behruz
Description:
    Loads trained Random Forest classifier to predict loan approval
    for new applicant profiles with confidence score.
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np


def load_model(model_path: str = "models/loan_approval_model.pkl"):
    """Loads serialized ML model."""
    if not os.path.exists(model_path):
        # Fallback to root model file if present
        if os.path.exists("loan_approval_model.pkl"):
            model_path = "loan_approval_model.pkl"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}. Run 'python src/train.py' first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_loan_approval(applicant_data: dict, model=None):
    """
    Given applicant profile attributes, predicts loan approval.

    Features expected:
    - Gender: 1 for Male, 0 for Female
    - Married: 1 for Yes, 0 for No
    - Dependents: 0, 1, 2, or 3
    - Education: 0 for Graduate, 1 for Not Graduate
    - Self_Employed: 0 for No, 1 for Yes
    - ApplicantIncome: numeric ($)
    - CoapplicantIncome: numeric ($)
    - LoanAmount: numeric (in thousands $)
    - Loan_Amount_Term: term in days (usually 360)
    - Credit_History: 1.0 (meets guidelines) or 0.0 (does not)
    - Property_Area: 0 for Rural, 1 for Semiurban, 2 for Urban
    """
    if model is None:
        model = load_model()

    feature_order = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    input_df = pd.DataFrame([applicant_data])[feature_order]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    status = "APPROVED" if prediction == 1 else "REJECTED"
    confidence = probabilities[prediction] * 100

    return {
        "status": status,
        "prediction": int(prediction),
        "confidence": confidence,
        "probabilities": {
            "rejection": probabilities[0] * 100,
            "approval": probabilities[1] * 100
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Loan Approval for an Applicant")
    parser.add_argument("--gender", type=int, default=1, choices=[0, 1], help="Gender (1=Male, 0=Female)")
    parser.add_argument("--married", type=int, default=1, choices=[0, 1], help="Married (1=Yes, 0=No)")
    parser.add_argument("--dependents", type=int, default=0, choices=[0, 1, 2, 3], help="Number of Dependents")
    parser.add_argument("--education", type=int, default=0, choices=[0, 1], help="Education (0=Graduate, 1=Not Graduate)")
    parser.add_argument("--self_employed", type=int, default=0, choices=[0, 1], help="Self Employed (1=Yes, 0=No)")
    parser.add_argument("--income", type=float, default=5400, help="Applicant Monthly Income ($)")
    parser.add_argument("--coapplicant_income", type=float, default=1600, help="Co-applicant Income ($)")
    parser.add_argument("--loan_amount", type=float, default=130, help="Loan Amount (in Thousands $)")
    parser.add_argument("--term", type=float, default=360, help="Loan Term (Days)")
    parser.add_argument("--credit_history", type=float, default=1.0, choices=[0.0, 1.0], help="Credit History (1=Good, 0=Bad)")
    parser.add_argument("--property_area", type=int, default=1, choices=[0, 1, 2], help="Property Area (0=Rural, 1=Semiurban, 2=Urban)")
    parser.add_argument("--demo", action="store_true", help="Run comparative demonstration with 2 sample profiles")
    args = parser.parse_args()

    model = load_model()

    if args.demo:
        print("\n" + "=" * 65)
        print(f"{'LOAN APPROVAL INFERENCE - COMPARATIVE DEMO':^65}")
        print("=" * 65)

        sample_approved = {
            'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': 0, 'Self_Employed': 0,
            'ApplicantIncome': 6500, 'CoapplicantIncome': 2200, 'LoanAmount': 140,
            'Loan_Amount_Term': 360, 'Credit_History': 1.0, 'Property_Area': 1
        }
        res1 = predict_loan_approval(sample_approved, model)
        print("\n[Profile 1: High Income, Good Credit History, Semiurban]")
        print(f" -> Decision   : [{res1['status']}]")
        print(f" -> Confidence : {res1['confidence']:.2f}% (Approval: {res1['probabilities']['approval']:.1f}%, Rejection: {res1['probabilities']['rejection']:.1f}%)")

        sample_rejected = {
            'Gender': 1, 'Married': 0, 'Dependents': 2, 'Education': 1, 'Self_Employed': 1,
            'ApplicantIncome': 2100, 'CoapplicantIncome': 0, 'LoanAmount': 220,
            'Loan_Amount_Term': 360, 'Credit_History': 0.0, 'Property_Area': 0
        }
        res2 = predict_loan_approval(sample_rejected, model)
        print("\n[Profile 2: Low Income, No Credit History, High Loan Amount]")
        print(f" -> Decision   : [{res2['status']}]")
        print(f" -> Confidence : {res2['confidence']:.2f}% (Approval: {res2['probabilities']['approval']:.1f}%, Rejection: {res2['probabilities']['rejection']:.1f}%)")
        print("\n" + "=" * 65)
        return

    applicant = {
        'Gender': args.gender,
        'Married': args.married,
        'Dependents': args.dependents,
        'Education': args.education,
        'Self_Employed': args.self_employed,
        'ApplicantIncome': args.income,
        'CoapplicantIncome': args.coapplicant_income,
        'LoanAmount': args.loan_amount,
        'Loan_Amount_Term': args.term,
        'Credit_History': args.credit_history,
        'Property_Area': args.property_area
    }

    result = predict_loan_approval(applicant, model)
    print("\n" + "=" * 50)
    print(f"{'APPLICANT EVALUATION RESULT':^50}")
    print("=" * 50)
    print(f"Decision          : {result['status']}")
    print(f"Confidence        : {result['confidence']:.2f}%")
    print(f"Approval Chance   : {result['probabilities']['approval']:.2f}%")
    print(f"Rejection Chance  : {result['probabilities']['rejection']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
