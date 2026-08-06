import cv2
import re
import json
import easyocr

# Load OCR model only once
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])
print("EasyOCR Loaded Successfully")


def extract_ocr(image_path):
    # Read Image
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Unable to read image.")

    # Perform OCR
    results = reader.readtext(image_path)

    # Store complete text
    full_text = ""

    for detection in results:
        text = detection[1]
        full_text += text + "\n"

   
    # Extract Required Information


    aadhaar = None
    dob = None
    gender = None
    name = None

    aadhaar_pattern = r"\d{4}\s\d{4}\s\d{4}"
    dob_pattern = r"\d{2}/\d{2}/\d{4}"

    match = re.search(aadhaar_pattern, full_text)
    if match:
        aadhaar = match.group()

    match = re.search(dob_pattern, full_text)
    if match:
        dob = match.group()

    if "Male" in full_text:
        gender = "Male"

    elif "Female" in full_text:
        gender = "Female"

    elif "Transgender" in full_text:
        gender = "Transgender"

    lines = [line.strip() for line in full_text.split("\n") if line.strip()]

    ignore_words = [
        "Government",
        "India",
        "Unique",
        "Identification",
        "Authority",
        "Address",
        "DOB",
        "Birth",
        "Year",
        "Male",
        "Female",
        "Transgender",
        "Aadhaar"
    ]

    for line in lines:

        if any(word.lower() in line.lower() for word in ignore_words):
            continue

        if len(line.split()) >= 2:

            if line.replace(" ", "").isalpha():
                name = line
                break


    # Validation


    errors = []

    if aadhaar:
        if len(aadhaar.replace(" ", "")) != 12:
            errors.append("Invalid Aadhaar Number")
    else:
        errors.append("Aadhaar Number Missing")

    if dob is None:
        errors.append("DOB Missing")

    if gender is None:
        errors.append("Gender Missing")

    if name is None:
        errors.append("Name Missing")


    # Verification Result


    if len(errors) == 0:
        ocr_verified = True
    else:
        ocr_verified = False

    # Final Data


    extracted_data = {

        "document_type": "Aadhaar",

        "name": name,

        "dob": dob,

        "gender": gender,

        "aadhaar_number": aadhaar,

        "validation_errors": errors
    }

    return extracted_data, ocr_verified