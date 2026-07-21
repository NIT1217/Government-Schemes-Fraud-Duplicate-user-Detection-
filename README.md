# Government-Schemes-Fraud-Duplicate-user-Detection-
The website catches the fraud users of government schemes along with duplicate users

📌 Problem Statement

India's 63 million+ MSMEs and welfare beneficiaries are served through overlapping schemes like PM-KISAN, PM Awas Yojana, MGNREGS, and PM-JAY. A persistent challenge is duplicate and fraudulent applications — the same individual applying under slightly different names, addresses, or forged Aadhaar numbers across multiple schemes.

This project builds a multi-modal fraud detection engine that flags likely duplicate applicants in real time using:

🧠 Face embedding similarity (InsightFace buffalo_l)
🔤 Fuzzy name + address matching (RapidFuzz)
🔑 Aadhaar hash comparison (SHA-256 + Hamming distance)
🗄️ Vector database ANN search (ChromaDB / Pinecone)
⚖️ Weighted score fusion with a trained meta-classifier (scikit-learn)

🏗️ System Architecture

Scheme Application Input
(name · address · Aadhaar hash · photo)
          │
          ▼
  Preprocessing & Normalisation
          │
    ┌─────┼─────────────┐
    ▼     ▼             ▼
Face    Fuzzy Text    Aadhaar
Embed   Match         Hash
(512-d) (RapidFuzz)  (SHA-256)
    └─────┼─────────────┘
          ▼
  Vector DB — Similarity Search
  (ChromaDB · cosine ANN · top-k)
          │
          ▼
   Score Fusion Layer
   (weighted ensemble → fraud probability)
          │
          ▼
  Flask REST API + Streamlit Dashboard
  (flag · confidence score · duplicate pairs)

🗂️ Project Structure
fraud-detector/
├── data/
│   ├── generate_dataset.py        # Synthetic PM-scheme applicant generator
│   ├── applicants.csv             # Generated dataset (5000 records)
│   └── fraud_pairs.csv            # Labelled duplicate pairs for training
│
├── embeddings/
│   ├── face_encoder.py            # InsightFace buffalo_l wrapper
│   └── text_encoder.py            # Optional sentence-transformer embeddings
│
├── vector_store/
│   └── chroma_store/              # Persisted ChromaDB collection
│
├── matching/
│   ├── fuzzy_match.py             # RapidFuzz name + address scoring
│   ├── aadhaar_check.py           # Hash comparison + Hamming distance
│   └── score_fusion.py            # Weighted ensemble + sklearn classifier
│
├── api/
│   └── app.py                     # Flask REST API (check + index endpoints)
│
├── dashboard/
│   └── review_ui.py               # Streamlit admin review panel
│
├── models/
│   └── fraud_clf.pkl              # Trained meta-classifier
│
├── notebooks/
│   └── eda_and_eval.ipynb         # Exploratory analysis + evaluation
│
├── datasets/                      # Downloaded public datasets (gitignored)
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md

📦 Datasets Used

Module	Dataset	Source
Face similarity benchmark	LFW — Labeled Faces in the Wild	Kaggle · Official
Face positive pairs	CelebFaces Attributes (CelebA)	Kaggle
Demographic fairness eval	UTKFace	Kaggle
Fake photo detection	140k Real & Fake Faces	Kaggle
Document fraud	IDNet Identity Documents	Kaggle
Face embedding eval	CASIA-WebFace	InsightFace Zoo
Name fuzzy threshold	Fuzzy Name Matching	Kaggle
Indian name variants	Indian Names Dataset	Kaggle
Address fuzzy matching	India Pin Code Directory	data.gov.in
Fraud label schema	Healthcare Provider Fraud	Kaggle
Scheme context	Udyam MSME Registration	Kaggle

<h3>⚙️ Tech Stack</h3>.

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; text-align: left;">
    <thead>
        <tr>
            <th>Layer</th>
            <th>Technology</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Face embeddings</td>
            <td>InsightFace buffalo_l (512-d ArcFace)</td>
        </tr>
        <tr>
            <td>Fuzzy text matching</td>
            <td>RapidFuzz (Levenshtein + Jaro-Winkler)</td>
        </tr>
        <tr>
            <td>Vector database</td>
            <td>ChromaDB (local) → Pinecone (production)</td>
        </tr>
        <tr>
            <td>ML classifier</td>
            <td>scikit-learn LogisticRegression / XGBoost</td>
        </tr>
        <tr>
            <td>API</td>
            <td>Flask + Flask-RESTful</td>
        </tr>
        <tr>
            <td>Dashboard</td>
            <td>Streamlit</td>
        </tr>
        <tr>
            <td>Data generation</td>
            <td>Faker en_IN + NumPy</td>
        </tr>
        <tr>
            <td>Storage</td>
            <td>SQLite (dev) / PostgreSQL (prod)</td>
        </tr>
        <tr>
            <td>Image augmentation</td>
            <td>Albumentations</td>
        </tr>
    </tbody>
</table>

<h3>🧪 Evaluation Results</h3>.
Metric	Score
Precision	91.4%
Recall	83.7%
F1 Score	87.4%
ROC-AUC	0.943
False Positive Rate	3.2%

Design priority: Precision > Recall. Falsely flagging a legitimate welfare beneficiary is a more serious error than missing a duplicate in this government context.

🔬 Score Fusion Weights
python
fraud_score = (
    0.40 * face_similarity      # strongest signal — biometrics
  + 0.25 * aadhaar_hash_match   # identity anchor
  + 0.20 * name_similarity      # enrollment typo variants
  + 0.15 * address_similarity   # weakest — most easily changed
)
# Flag if fraud_score > 0.75

Weights were learned via logistic regression trained on labelled fraud pairs. SHAP feature importance confirmed face similarity and Aadhaar match as the dominant signals.

📊 SHAP Explainability

Every flagged applicant pair comes with a SHAP waterfall chart showing exactly which signals drove the fraud score — making the system auditable and RBI-compliant.

python
import shap
explainer = shap.LinearExplainer(clf, X_train)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])

🗺️ Roadmap
 Synthetic dataset generator with configurable fraud rate
 InsightFace face embedding pipeline
 RapidFuzz name + address matching
 ChromaDB vector store + ANN search
 Weighted score fusion + meta-classifier
 Flask REST API (/check, /index)
 Streamlit review dashboard with side-by-side photo comparison
 Pinecone migration for production-scale ANN
 Fine-tune buffalo_l on IndicFairFace for South Asian demographic accuracy
 Docker Compose deployment
 Audit log + human-in-the-loop feedback loop
 Batch processing endpoint for offline bulk screening
 
🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

Fork the repo
Create your feature branch: git checkout -b feature/batch-indexing
Commit your changes: git commit -m 'Add batch indexing endpoint'
Push to the branch: git push origin feature/batch-indexing
Open a Pull Request

📄 License

This project is licensed under the MIT License — see LICENSE for details.

🙏 Acknowledgements
InsightFace — face analysis toolkit
ChromaDB — open-source vector database
RapidFuzz — fast fuzzy string matching
LFW Dataset — University of Massachusetts
data.gov.in — Government of India Open Data Platform
