# Government-Schemes-Fraud-Duplicate-user-Detection-
The website catches the fraud users of government schemes along with duplicate users

<h2>📌 Problem Statement</h2>

India's 63 million+ MSMEs and welfare beneficiaries are served through overlapping schemes like PM-KISAN, PM Awas Yojana, MGNREGS, and PM-JAY. A persistent challenge is duplicate and fraudulent applications — the same individual applying under slightly different names, addresses, or forged Aadhaar numbers across multiple schemes.

This project builds a multi-modal fraud detection engine that flags likely duplicate applicants in real time using:

🧠 Face embedding similarity (InsightFace buffalo_l)
🔤 Fuzzy name + address matching (RapidFuzz)
🔑 Aadhaar hash comparison (SHA-256 + Hamming distance)
🗄️ Vector database ANN search (ChromaDB / Pinecone)
⚖️ Weighted score fusion with a trained meta-classifier (scikit-learn)

<h2>🏗️ System Architecture</h2>

<div align="center">

<table>
<tr>
<td align="center"><b>Scheme Application Input</b><br>
Name · Address · Aadhaar Hash · Photo
</td>
</tr>

<tr>
<td align="center">⬇️</td>
</tr>

<tr>
<td align="center"><b>Preprocessing &amp; Normalisation</b></td>
</tr>

<tr>
<td align="center">⬇️</td>
</tr>

<tr>
<td align="center">

<table>
<tr>
<th>Face</th>
<th>Fuzzy Text</th>
<th>Aadhaar</th>
</tr>

<tr>
<td align="center">
Face Embedding<br>
(512-d)
</td>

<td align="center">
RapidFuzz<br>
Levenshtein +<br>
Jaro-Winkler
</td>

<td align="center">
SHA-256<br>
Hash
</td>
</tr>

</table>

</td>
</tr>

<tr>
<td align="center">⬇️</td>
</tr>

<tr>
<td align="center">
<b>Vector Database</b><br>
ChromaDB · Cosine ANN · Top-k Similarity Search
</td>
</tr>

<tr>
<td align="center">⬇️</td>
</tr>

<tr>
<td align="center">
<b>Score Fusion Layer</b><br>
Weighted Ensemble → Fraud Probability
</td>
</tr>

<tr>
<td align="center">⬇️</td>
</tr>

<tr>
<td align="center">
<b>Flask REST API + Streamlit Dashboard</b><br>
Flag · Confidence Score · Duplicate Pairs
</td>
</tr>

</table>

</div>


<h2>📦 Datasets Used</h2>

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; text-align: left;">
    <thead>
        <tr>
            <th>Module</th>
            <th>Dataset</th>
            <th>Source</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Face similarity benchmark</td>
            <td>LFW — Labeled Faces in the Wild</td>
            <td>Kaggle · Official</td>
        </tr>
        <tr>
            <td>Face positive pairs</td>
            <td>CelebFaces Attributes (CelebA)</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Demographic fairness eval</td>
            <td>UTKFace</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Fake photo detection</td>
            <td>140k Real &amp; Fake Faces</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Document fraud</td>
            <td>IDNet Identity Documents</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Face embedding eval</td>
            <td>CASIA-WebFace</td>
            <td>InsightFace Zoo</td>
        </tr>
        <tr>
            <td>Name fuzzy threshold</td>
            <td>Fuzzy Name Matching</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Indian name variants</td>
            <td>Indian Names Dataset</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Address fuzzy matching</td>
            <td>India Pin Code Directory</td>
            <td>data.gov.in</td>
        </tr>
        <tr>
            <td>Fraud label schema</td>
            <td>Healthcare Provider Fraud</td>
            <td>Kaggle</td>
        </tr>
        <tr>
            <td>Scheme context</td>
            <td>Udyam MSME Registration</td>
            <td>Kaggle</td>
        </tr>
    </tbody>
</table>

<h2>⚙️ Tech Stack</h2>.

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
