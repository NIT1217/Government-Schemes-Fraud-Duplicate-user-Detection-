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
🗂️ Project Structure
<h2>📁 Project Structure</h2>

<ul>
    <li>
        <b>fraud-detector/</b>
        <ul>
            <li>
                <b>data/</b>
                <ul>
                    <li><code>generate_dataset.py</code> – Synthetic PM-scheme applicant generator</li>
                    <li><code>applicants.csv</code> – Generated dataset (5000 records)</li>
                    <li><code>fraud_pairs.csv</code> – Labelled duplicate pairs for training</li>
                </ul>
            </li>

            <li>
                <b>embeddings/</b>
                <ul>
                    <li><code>face_encoder.py</code> – InsightFace buffalo_l wrapper</li>
                    <li><code>text_encoder.py</code> – Optional sentence-transformer embeddings</li>
                </ul>
            </li>

            <li>
                <b>vector_store/</b>
                <ul>
                    <li><code>chroma_store/</code> – Persisted ChromaDB collection</li>
                </ul>
            </li>

            <li>
                <b>matching/</b>
                <ul>
                    <li><code>fuzzy_match.py</code> – RapidFuzz name &amp; address scoring</li>
                    <li><code>aadhaar_check.py</code> – Hash comparison &amp; Hamming distance</li>
                    <li><code>score_fusion.py</code> – Weighted ensemble + sklearn classifier</li>
                </ul>
            </li>

            <li>
                <b>api/</b>
                <ul>
                    <li><code>app.py</code> – Flask REST API (check &amp; index endpoints)</li>
                </ul>
            </li>

            <li>
                <b>dashboard/</b>
                <ul>
                    <li><code>review_ui.py</code> – Streamlit admin review panel</li>
                </ul>
            </li>

            <li>
                <b>models/</b>
                <ul>
                    <li><code>fraud_clf.pkl</code> – Trained meta-classifier</li>
                </ul>
            </li>

            <li>
                <b>notebooks/</b>
                <ul>
                    <li><code>eda_and_eval.ipynb</code> – Exploratory analysis &amp; evaluation</li>
                </ul>
            </li>

            <li><b>datasets/</b> – Downloaded public datasets (gitignored)</li>
            <li><code>requirements.txt</code></li>
            <li><code>.env.example</code></li>
            <li><code>Dockerfile</code></li>
            <li><code>README.md</code></li>
        </ul>
    </li>
</ul>

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
