#!/usr/bin/env python3
"""
ResumeCraft — Evaluation Script
================================
Usage:
python resumecraft_eval.py --csv /path/to/data.csv --output-dir ./artifacts
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load

import torch
from torch.utils.data import DataLoader
from resumecraft_train import DualEncoderFusion, ResumeJobDataset, Collator, mean_pooling
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer


# -----------------------
# Arguments
# -----------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./artifacts")
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

# -----------------------
# Load config + TF-IDF
# -----------------------
config_path = os.path.join(args.output_dir, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

vectorizer = load(os.path.join(args.output_dir, "tfidf_vectorizer.joblib"))
svd = load(os.path.join(args.output_dir, "tfidf_svd.joblib"))

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(args.csv)
job_texts = df[config["text_cols"][0]].fillna("").astype(str).tolist()
resume_texts = df[config["text_cols"][1]].fillna("").astype(str).tolist()
labels = df[config["label_col"]].astype(float).values
labels = np.clip(labels, 0.0, 1.0)

# TF-IDF -> SVD
corpus = job_texts + resume_texts
X = vectorizer.transform(corpus)
X_svd = svd.transform(X)
tfidf_job = X_svd[:len(job_texts)]
tfidf_resume = X_svd[len(job_texts):]

dataset = ResumeJobDataset(resume_texts, job_texts, labels, tfidf_resume, tfidf_job)
resume_tokenizer = AutoTokenizer.from_pretrained(config["resume_model"])
job_tokenizer = AutoTokenizer.from_pretrained(config["job_model"])
collator = Collator(resume_tokenizer, job_tokenizer,
                    max_len_resume=config["max_len_resume"],
                    max_len_job=config["max_len_job"])

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

# -----------------------
# Load model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualEncoderFusion(
    resume_model_name=config["resume_model"],
    job_model_name=config["job_model"],
    tfidf_dim=config["svd_dim"],
    sbert_hidden_dim=384,
    hidden_dim=256,
).to(device)

model.load_state_dict(torch.load(os.path.join(args.output_dir, "fusion_head.pt"), map_location=device))
model.eval()

# -----------------------
# Evaluate
# -----------------------
preds_all = []
labels_all = []

with torch.no_grad():
    for r_inputs, j_inputs, tfidf_r, tfidf_j, batch_labels in loader:
        r_inputs = {k: v.to(device) for k,v in r_inputs.items()}
        j_inputs = {k: v.to(device) for k,v in j_inputs.items()}
        tfidf_r = tfidf_r.to(device)
        tfidf_j = tfidf_j.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(r_inputs, j_inputs, tfidf_r, tfidf_j)
        batch_preds = torch.sigmoid(logits)
        preds_all.append(batch_preds.cpu().numpy())
        labels_all.append(batch_labels.cpu().numpy())

preds_all = np.concatenate(preds_all)
labels_all = np.concatenate(labels_all)

mae = mean_absolute_error(labels_all, preds_all)
rmse = np.sqrt(mean_squared_error(labels_all, preds_all))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# -----------------------
# Top 5 best / worst matches
# -----------------------
df["predicted_score"] = preds_all
top5 = df.nlargest(5, "predicted_score")
bottom5 = df.nsmallest(5, "predicted_score")

print("\nTop 5 predicted matches:")
print(top5[[config["text_cols"][1], config["text_cols"][0], "predicted_score"]])

print("\nBottom 5 predicted matches:")
print(bottom5[[config["text_cols"][1], config["text_cols"][0], "predicted_score"]])
