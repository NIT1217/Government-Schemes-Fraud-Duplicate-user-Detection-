from model.final_score import calculate_fraud_score

def prediction(selfie_image, document_image):
    return calculate_fraud_score(selfie_image, document_image)
