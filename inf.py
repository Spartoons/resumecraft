import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from joblib import load

from resumecraft_train import DualEncoderFusion, Collator, mean_pooling

# -------------------------
# Config
# -------------------------
CSV_PATH = "10000jobs_candidates/resume_job_matching_dataset.csv"
ARTIFACT_DIR = "./artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 20
LABEL_SCALE = 5.0  # your 1-5 scale

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv(CSV_PATH)
job_texts = df["job_description"].fillna("").astype(str).tolist()
resume_texts = df["resume"].fillna("").astype(str).tolist()
labels = df["match_score"].astype(float).values / LABEL_SCALE  # normalize to 0-1

# -------------------------
# Load TF-IDF + SVD
# -------------------------
tfidf_vectorizer = load(f"{ARTIFACT_DIR}/tfidf_vectorizer.joblib")
svd = load(f"{ARTIFACT_DIR}/tfidf_svd.joblib")

X = tfidf_vectorizer.transform(job_texts + resume_texts)
X_svd = svd.transform(X)
tfidf_job = X_svd[:len(job_texts)]
tfidf_resume = X_svd[len(job_texts):]

# -------------------------
# Load model
# -------------------------
resume_model_name = job_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # or from config.json
model = DualEncoderFusion(
    resume_model_name=resume_model_name,
    job_model_name=job_model_name,
    tfidf_dim=tfidf_job.shape[1],
).to(DEVICE)
model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/fusion_head.pt", map_location=DEVICE))
model.eval()

# -------------------------
# Tokenizers + Collator
# -------------------------
resume_tokenizer = AutoTokenizer.from_pretrained(resume_model_name)
job_tokenizer = AutoTokenizer.from_pretrained(job_model_name)
collator = Collator(resume_tokenizer, job_tokenizer)

# -------------------------
# Sample 20 random indices
# -------------------------
indices = random.sample(range(len(labels)), N_SAMPLES)
sample_resumes = [resume_texts[i] for i in indices]
sample_jobs = [job_texts[i] for i in indices]
sample_labels = labels[indices]
sample_tfidf_r = torch.tensor(tfidf_resume[indices], dtype=torch.float32).to(DEVICE)
sample_tfidf_j = torch.tensor(tfidf_job[indices], dtype=torch.float32).to(DEVICE)

# -------------------------
# Tokenize
# -------------------------
r_inputs = resume_tokenizer(sample_resumes, padding=True, truncation=True, max_length=256, return_tensors="pt")
j_inputs = job_tokenizer(sample_jobs, padding=True, truncation=True, max_length=512, return_tensors="pt")
r_inputs = {k: v.to(DEVICE) for k, v in r_inputs.items()}
j_inputs = {k: v.to(DEVICE) for k, v in j_inputs.items()}

# -------------------------
# Predict
# -------------------------
with torch.no_grad():
    logits = model(r_inputs, j_inputs, sample_tfidf_r, sample_tfidf_j)
    preds = torch.sigmoid(logits).cpu().numpy() * LABEL_SCALE  # scale back to 1-5

sample_labels = sample_labels * LABEL_SCALE  # scale back

# -------------------------
# Print results
# -------------------------
print("Predicted vs Expected:")
for p, l in zip(preds, sample_labels):
    print(f"{p:.2f}  <--->  {l:.1f}")
