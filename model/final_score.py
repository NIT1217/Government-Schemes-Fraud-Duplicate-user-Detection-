import json

from model.ocr import extract_ocr
from model.verify_face import verify_face
from model.liveness import check_liveness
from model.duplicate import check_duplicate


# Fraud Score Calculator

def calculate_fraud_score(selfie_path, document_path):
    print("RUNNING IDENTITY VERIFICATION MODULES")
 
    # OCR Verification
    ocr_data, ocr_verified = extract_ocr(document_path)

    # Face Verification
    face_similarity, face_verified = verify_face(
        selfie_path,
        document_path
    )
    
    # Liveness Detection
    liveness_score, liveness_verified = check_liveness(
        selfie_path
    )

    # Duplicate Detection
    duplicate_found, duplicate_similarity, best_match = check_duplicate(
        selfie_path
    )


    # Fraud Score Calculation
 

    fraud_score = 0

    # OCR (20 Marks)
    if ocr_verified:
        fraud_score += 20

    # Face Match (35 Marks)
    if face_verified:
        fraud_score += 35

    # Liveness (25 Marks)
    if liveness_verified:
        fraud_score += 25

    # Duplicate Detection (20 Marks)
    if not duplicate_found:
        fraud_score += 20

    
    # Risk Level

    if fraud_score >= 90:

        risk_level = "LOW"

        decision = "APPROVED"

    elif fraud_score >= 70:

        risk_level = "MEDIUM"

        decision = "MANUAL REVIEW"

    else:

        risk_level = "HIGH"

        decision = "REJECTED"

    
    # Final Report

    report = {
        "face_verified": face_verified,
        "liveness_verified": liveness_verified,
        "duplicate_found": duplicate_found,
        "matched_person": best_match,
        "fraud_score": fraud_score,
        "risk_level": risk_level,
    }

    return report
