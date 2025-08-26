#!/usr/bin/env python3
"""
ResumeCraft — Dual-Encoder SBERT + TF‑IDF Fusion (Training Script)
==================================================================

Overview
--------
- Inputs: CSV with columns **job_description**, **resume**, **match_score**.
- Architecture: Two separate transformer encoders (resume encoder & job encoder) +
  sparse TF‑IDF features reduced with TruncatedSVD. A small MLP fuses:
    • cosine(sim_tfidf), cosine(sim_sbert),
    • dense TF‑IDF projections for resume & job,
    • pooled SBERT embeddings for resume & job,
    and predicts a scalar **match score** in [0, 1].
- Training: End‑to‑end fine‑tuning of both encoders and the fusion head.
  TF‑IDF features are fixed (non‑differentiable).

Quick Start
-----------
1) Install deps (Python 3.9+ recommended):
   pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -U transformers sentencepiece accelerate
   pip install -U scikit-learn pandas numpy joblib tqdm

2) Run training:
   python resumecraft_train.py --csv /path/to/dataset.csv --text-cols job_description resume --label-col match_score \
       --output-dir ./artifacts --epochs 3 --batch-size 8

3) Artifacts saved to --output-dir:
   - tfidf_vectorizer.joblib
   - tfidf_svd.joblib
   - fusion_head.pt
   - resume_encoder.pt / job_encoder.pt (state dicts)
   - config.json (model/tokenizer names, dims, etc.)

Notes
-----
- If your match_score is in [0,100], pass --label-scale 100 to normalize to [0,1].
- Start small (MiniLM) and CPU if needed; switch to GPU with `accelerate launch` or CUDA.
- This script is intentionally simple & readable as a solid starting point.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from joblib import dump

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
    summed = torch.sum(last_hidden_state * mask, dim=1)             # (B, H)
    counts = torch.clamp(mask.sum(dim=1), min=1e-6)                 # (B, 1)
    return summed / counts


# ------------------------------
# Dataset
# ------------------------------
class ResumeJobDataset(Dataset):
    def __init__(self,
                 resume_texts: List[str],
                 job_texts: List[str],
                 labels: np.ndarray,
                 tfidf_resume: np.ndarray,
                 tfidf_job: np.ndarray):
        assert len(resume_texts) == len(job_texts) == len(labels) == tfidf_resume.shape[0] == tfidf_job.shape[0]
        self.resume_texts = resume_texts
        self.job_texts = job_texts
        self.labels = labels.astype(np.float32)
        self.tfidf_resume = tfidf_resume.astype(np.float32)
        self.tfidf_job = tfidf_job.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "resume_text": self.resume_texts[idx],
            "job_text": self.job_texts[idx],
            "label": self.labels[idx],
            "tfidf_resume": self.tfidf_resume[idx],
            "tfidf_job": self.tfidf_job[idx],
        }


# ------------------------------
# Model
# ------------------------------
class DualEncoderFusion(nn.Module):
    def __init__(self,
                 resume_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 job_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 tfidf_dim: int = 300,
                 sbert_hidden_dim: int = 384,  # MiniLM-L6-v2 pooled dim
                 hidden_dim: int = 256):
        super().__init__()
        # Two independent encoders
        self.resume_encoder = AutoModel.from_pretrained(resume_model_name)
        self.job_encoder = AutoModel.from_pretrained(job_model_name)

        self.resume_model_name = resume_model_name
        self.job_model_name = job_model_name

        self.tfidf_dim = tfidf_dim
        self.sbert_hidden_dim = sbert_hidden_dim

        # Fusion head inputs:
        # sim_tfidf (1) + sim_sbert (1) + tfidf_r (D) + tfidf_j (D) + sbert_r (H) + sbert_j (H)
        fusion_in = 1 + 1 + (tfidf_dim * 2) + (sbert_hidden_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # output logit
        )

    def forward(self,
                resume_inputs: Dict[str, torch.Tensor],
                job_inputs: Dict[str, torch.Tensor],
                tfidf_resume: torch.Tensor,
                tfidf_job: torch.Tensor) -> torch.Tensor:
        # Encode resume
        r_out = self.resume_encoder(**resume_inputs)
        r_pooled = mean_pooling(r_out.last_hidden_state, resume_inputs["attention_mask"])  # (B, H)

        # Encode job
        j_out = self.job_encoder(**job_inputs)
        j_pooled = mean_pooling(j_out.last_hidden_state, job_inputs["attention_mask"])      # (B, H)

        # Normalize for cosine similarity
        r_norm = F.normalize(r_pooled, p=2, dim=-1)
        j_norm = F.normalize(j_pooled, p=2, dim=-1)
        sim_sbert = torch.sum(r_norm * j_norm, dim=-1, keepdim=True)                        # (B, 1)

        # TF-IDF cosine similarity
        tfidf_r_norm = F.normalize(tfidf_resume, p=2, dim=-1)
        tfidf_j_norm = F.normalize(tfidf_job, p=2, dim=-1)
        sim_tfidf = torch.sum(tfidf_r_norm * tfidf_j_norm, dim=-1, keepdim=True)            # (B, 1)

        # Concatenate features
        feat = torch.cat([sim_tfidf, sim_sbert, tfidf_resume, tfidf_job, r_pooled, j_pooled], dim=-1)
        logit = self.fusion(feat).squeeze(-1)                                               # (B,)
        return logit


# ------------------------------
# Collate
# ------------------------------
@dataclass
class Collator:
    resume_tokenizer: AutoTokenizer
    job_tokenizer: AutoTokenizer
    max_len_resume: int = 256
    max_len_job: int = 512

    def __call__(self, batch):
        resume_texts = [b["resume_text"] for b in batch]
        job_texts = [b["job_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        tfidf_resume = torch.tensor(np.stack([b["tfidf_resume"] for b in batch], axis=0), dtype=torch.float32)
        tfidf_job = torch.tensor(np.stack([b["tfidf_job"] for b in batch], axis=0), dtype=torch.float32)

        r_inputs = self.resume_tokenizer(
            resume_texts, padding=True, truncation=True, max_length=self.max_len_resume, return_tensors="pt"
        )
        j_inputs = self.job_tokenizer(
            job_texts, padding=True, truncation=True, max_length=self.max_len_job, return_tensors="pt"
        )
        return r_inputs, j_inputs, tfidf_resume, tfidf_job, labels


# ------------------------------
# Training / Eval
# ------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, loss_type="mse"):
    model.train()
    total_loss = 0.0
    for r_inputs, j_inputs, tfidf_r, tfidf_j, labels in tqdm(loader, desc="Train", leave=False):
        r_inputs = {k: v.to(device) for k, v in r_inputs.items()}
        j_inputs = {k: v.to(device) for k, v in j_inputs.items()}
        tfidf_r = tfidf_r.to(device)
        tfidf_j = tfidf_j.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(r_inputs, j_inputs, tfidf_r, tfidf_j)
        if loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:  # mse with sigmoid output target in [0,1]
            preds = torch.sigmoid(logits)
            loss = F.mse_loss(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_type="mse"):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for r_inputs, j_inputs, tfidf_r, tfidf_j, labels in tqdm(loader, desc="Eval", leave=False):
        r_inputs = {k: v.to(device) for k, v in r_inputs.items()}
        j_inputs = {k: v.to(device) for k, v in j_inputs.items()}
        tfidf_r = tfidf_r.to(device)
        tfidf_j = tfidf_j.to(device)
        labels = labels.to(device)

        logits = model(r_inputs, j_inputs, tfidf_r, tfidf_j)
        if loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            probs = torch.sigmoid(logits)
        else:
            probs = torch.sigmoid(logits)
            loss = F.mse_loss(probs, labels)

        total_loss += loss.item() * labels.size(0)
        preds_all.append(probs.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    mae = float(np.mean(np.abs(preds_all - labels_all)))
    rmse = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))
    return total_loss / len(loader.dataset), mae, rmse


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with columns job_description,resume,match_score")
    parser.add_argument("--text-cols", nargs=2, default=["job_description", "resume"], help="Order: job_col resume_col")
    parser.add_argument("--label-col", type=str, default="match_score")
    parser.add_argument("--label-scale", type=float, default=None, help="If labels are 0-100, set 100 to normalize to 0-1")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--resume-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--job-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-len-resume", type=int, default=256)
    parser.add_argument("--max-len-job", type=int, default=512)

    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--svd-dim", type=int, default=300)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)

    parser.add_argument("--loss-type", type=str, choices=["mse", "bce"], default="mse",
                        help="Use 'mse' for continuous scores in [0,1]; 'bce' for binary labels {0,1}")

    parser.add_argument("--output-dir", type=str, default="./artifacts")
    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project="resumecraft",
        config=vars(args),
        name="dual_encoder_fusion",
        reinit=True
    )


    # Load data
    df = pd.read_csv(args.csv)
    job_col, resume_col = args.text_cols
    label_col = args.label_col
    assert job_col in df.columns and resume_col in df.columns and label_col in df.columns,         f"Expected columns {job_col}, {resume_col}, {label_col} in CSV"

    labels = df[label_col].astype(float).values
    if args.label_scale is not None and args.label_scale > 0:
        labels = labels / args.label_scale
    labels = np.clip(labels, 0.0, 1.0)

    job_texts = df[job_col].fillna("").astype(str).tolist()
    resume_texts = df[resume_col].fillna("").astype(str).tolist()

    # TF-IDF + SVD
    corpus = job_texts + resume_texts
    vectorizer = TfidfVectorizer(max_features=args.tfidf_max_features, ngram_range=(1,2), lowercase=True)
    X = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.random_seed)
    X_svd = svd.fit_transform(X)

    tfidf_job = X_svd[:len(job_texts)]
    tfidf_resume = X_svd[len(job_texts):]

    assert tfidf_job.shape[0] == tfidf_resume.shape[0] == len(labels)

    # Train/Val split
    idx = np.arange(len(labels))
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=args.random_seed)

    tr_dataset = ResumeJobDataset(
        [resume_texts[i] for i in tr_idx],
        [job_texts[i] for i in tr_idx],
        labels[tr_idx],
        tfidf_resume[tr_idx],
        tfidf_job[tr_idx],
    )
    va_dataset = ResumeJobDataset(
        [resume_texts[i] for i in va_idx],
        [job_texts[i] for i in va_idx],
        labels[va_idx],
        tfidf_resume[va_idx],
        tfidf_job[va_idx],
    )

    # Tokenizers
    resume_tokenizer = AutoTokenizer.from_pretrained(args.resume_model)
    job_tokenizer = AutoTokenizer.from_pretrained(args.job_model)

    collator = Collator(resume_tokenizer, job_tokenizer, args.max_len_resume, args.max_len_job)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    va_loader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderFusion(
        resume_model_name=args.resume_model,
        job_model_name=args.job_model,
        tfidf_dim=args.svd_dim,
        sbert_hidden_dim=384,
        hidden_dim=256,
    ).to(device)

    # Optimizer / Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    total_steps = len(tr_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    wandb.watch(model, log="all", log_freq=100)

    # Train
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, device, loss_type=args.loss_type)
        val_loss, mae, rmse = evaluate(model, va_loader, device, loss_type=args.loss_type)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": mae,
            "val_RMSE": rmse,
        })


        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "fusion_head.pt"))
            torch.save(model.resume_encoder.state_dict(), os.path.join(args.output_dir, "resume_encoder.pt"))
            torch.save(model.job_encoder.state_dict(), os.path.join(args.output_dir, "job_encoder.pt"))
        
        

    # Save TF-IDF artifacts + config
    dump(vectorizer, os.path.join(args.output_dir, "tfidf_vectorizer.joblib"))
    dump(svd, os.path.join(args.output_dir, "tfidf_svd.joblib"))

    config = {
        "resume_model": args.resume_model,
        "job_model": args.job_model,
        "svd_dim": args.svd_dim,
        "tfidf_max_features": args.tfidf_max_features,
        "max_len_resume": args.max_len_resume,
        "max_len_job": args.max_len_job,
        "loss_type": args.loss_type,
        "label_col": args.label_col,
        "text_cols": args.text_cols,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
