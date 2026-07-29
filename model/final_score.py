import json

from ocr import extract_ocr
from verify_face import verify_face
from liveness import check_liveness
from duplicate import check_duplicate


# -------------------------------------------------
# Fraud Score Calculator
# -------------------------------------------------

def calculate_fraud_score(selfie_path, document_path):

    print("\n======================================")
    print("RUNNING IDENTITY VERIFICATION MODULES")
    print("======================================")

    # ----------------------------------
    # OCR Verification
    # ----------------------------------

    ocr_data, ocr_verified = extract_ocr(document_path)

    # ----------------------------------
    # Face Verification
    # ----------------------------------

    face_similarity, face_verified = verify_face(
        selfie_path,
        document_path
    )

    # ----------------------------------
    # Liveness Detection
    # ----------------------------------

    liveness_score, liveness_verified = check_liveness(
        selfie_path
    )

    # ----------------------------------
    # Duplicate Detection
    # ----------------------------------

    duplicate_found, duplicate_similarity, best_match = check_duplicate(
        selfie_path
    )

    # ----------------------------------
    # Fraud Score Calculation
    # ----------------------------------

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

    # ----------------------------------
    # Risk Level
    # ----------------------------------

    if fraud_score >= 90:

        risk_level = "LOW"

        decision = "APPROVED"

    elif fraud_score >= 70:

        risk_level = "MEDIUM"

        decision = "MANUAL REVIEW"

    else:

        risk_level = "HIGH"

        decision = "REJECTED"

    # ----------------------------------
    # Final Report
    # ----------------------------------

    report = {

        "ocr_verified": ocr_verified,

        "face_verified": face_verified,

        "liveness_verified": liveness_verified,

        "duplicate_found": duplicate_found,

        "face_similarity": round(face_similarity, 4),

        "duplicate_similarity": round(duplicate_similarity, 4),

        "liveness_score": round(liveness_score, 4),

        "fraud_score": fraud_score,

        "risk_level": risk_level,

        "decision": decision,

        "matched_person": best_match,

        "ocr_data": ocr_data

    }

    return report


# -------------------------------------------------
# Testing
# -------------------------------------------------

if __name__ == "__main__":

    SELFIE_IMAGE = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\testdata\Balvinder PAN.jpg"

    DOCUMENT_IMAGE = r"C:\Users\HP\Desktop\Government Scheme Application\Government-Schemes-Fraud-Duplicate-user-Detection-\testdata\Balvinder PAN.jpg"

    report = calculate_fraud_score(

        SELFIE_IMAGE,

        DOCUMENT_IMAGE

    )
   
    print("=" * 60)
    print("FINAL FRAUD DETECTION REPORT")
    print("=" * 60)
    

    print(json.dumps(report, indent=4))