"""Train the two-head poverty model and log everything to TensorBoard.

Placeholder run (proves the pipeline end to end):
    python train.py --epochs 1 --run-name placeholder_1epoch
Then:  tensorboard --logdir runs
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import config as C
from features import Preprocessor, make_split
from model import PovertyNet


def recall_at_precision(y, p, w, target=0.7):
    prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


@torch.no_grad()
def predict(model, x_num, x_cat, device, bs=2048):
    model.eval()
    logits, regs = [], []
    for i in range(0, len(x_num), bs):
        l, r = model(x_num[i:i + bs].to(device), x_cat[i:i + bs].to(device))
        logits.append(l.cpu()); regs.append(r.cpu())
    return torch.cat(logits), torch.cat(regs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.TABLE_PATH))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lambda-reg", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    a = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(a.data)
    tr_idx, va_idx = make_split(df, a.fold)
    tr, va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    pre = Preprocessor().fit(tr)
    print(f"train={len(tr):,} val={len(va):,}  numeric={pre.n_numeric}  categorical={len(pre.cat_cols)}")

    def tensors(d):
        xn, xc = pre.transform(d)
        y = torch.tensor(d[C.POVERTY].to_numpy(), dtype=torch.float32)
        w = torch.tensor((d[C.WEIGHT] / d[C.WEIGHT].mean()).to_numpy(), dtype=torch.float32)
        return torch.from_numpy(xn), torch.from_numpy(xc), y, w

    xn_tr, xc_tr, y_tr, w_tr = tensors(tr)
    xn_va, xc_va, y_va, w_va = tensors(va)
    c_mu, c_sd = np.log(tr[C.CONSUMPTION].clip(lower=1)).mean(), np.log(tr[C.CONSUMPTION].clip(lower=1)).std()
    reg_tr = torch.tensor(((np.log(tr[C.CONSUMPTION].clip(lower=1)) - c_mu) / c_sd).to_numpy(), dtype=torch.float32)
    reg_va = torch.tensor(((np.log(va[C.CONSUMPTION].clip(lower=1)) - c_mu) / c_sd).to_numpy(), dtype=torch.float32)

    pos_w = float((1 - y_tr.mean()) / y_tr.mean())
    model = PovertyNet(pre.n_numeric, pre.cardinalities, dropout=a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    loader = DataLoader(TensorDataset(xn_tr, xc_tr, y_tr, w_tr, reg_tr), batch_size=a.batch_size, shuffle=True)

    writer = SummaryWriter(f"runs/{a.run_name}")
    cfg = {**vars(a), "n_train": len(tr), "n_val": len(va), "pos_weight": pos_w,
           "numeric_features": pre.num_cols, "categorical_features": pre.cat_cols}
    writer.add_text("config", "```\n" + json.dumps(cfg, indent=2, default=str) + "\n```")
    try:
        writer.add_graph(model, (xn_tr[:8].to(device), xc_tr[:8].to(device)))
    except Exception as e:  # graph tracing is nice-to-have
        print(f"[warn] add_graph failed: {e}")

    best, bad = -1.0, 0
    best_metrics = {}
    for epoch in range(a.epochs):
        model.train()
        tot = {"loss": 0.0, "poverty": 0.0, "consumption": 0.0}; n = 0
        for b, (xn, xc, y, w, r) in enumerate(loader):
            if a.max_batches and b >= a.max_batches:
                break
            xn, xc, y, w, r = (t.to(device) for t in (xn, xc, y, w, r))
            logit, reg = model(xn, xc)
            cw = torch.where(y == 1, torch.tensor(pos_w, device=device), torch.tensor(1.0, device=device))
            l_pov = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w * cw).mean()
            l_reg = F.huber_loss(reg, r)
            loss = l_pov + a.lambda_reg * l_reg
            opt.zero_grad(); loss.backward(); opt.step()
            tot["loss"] += loss.item(); tot["poverty"] += l_pov.item(); tot["consumption"] += l_reg.item(); n += 1
        sched.step()

        logit, reg = predict(model, xn_va, xc_va, device)
        prob = torch.sigmoid(logit).numpy()
        y_np, w_np = y_va.numpy(), w_va.numpy()
        pr_auc = float(average_precision_score(y_np, prob, sample_weight=w_np))
        rec = recall_at_precision(y_np, prob, w_np)
        val_reg = float(F.huber_loss(reg, reg_va))

        for k, v in tot.items():
            writer.add_scalar(f"loss/train_{k}", v / max(n, 1), epoch)
        writer.add_scalar("loss/val_consumption", val_reg, epoch)
        writer.add_scalar("val/pr_auc", pr_auc, epoch)
        writer.add_scalar("val/recall_at_precision_0.7", rec, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        for name, p in model.named_parameters():
            writer.add_histogram(f"weights/{name}", p.detach().cpu(), epoch)
            if p.grad is not None:
                writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), epoch)
        print(f"epoch {epoch + 1}/{a.epochs}  train_loss={tot['loss'] / max(n, 1):.4f}  "
              f"val PR-AUC={pr_auc:.4f}  recall@P0.7={rec:.4f}")

        if pr_auc > best:
            best, bad = pr_auc, 0
            best_metrics = {"hparam/val_pr_auc": pr_auc, "hparam/recall_at_p0.7": rec}
            torch.save({"state": model.state_dict(), "config": cfg}, f"runs/{a.run_name}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    writer.add_pr_curve("val/pr_curve", torch.tensor(y_np), torch.tensor(prob), global_step=0)
    # Embedding projector: first categorical column named like a geography, else the first one.
    if pre.cat_cols:
        geo = next((i for i, c in enumerate(pre.cat_cols) if any(k in c.lower() for k in ("district", "province"))), 0)
        labels = ["other"] + [k for k, _ in sorted(pre.vocab[pre.cat_cols[geo]].items(), key=lambda kv: kv[1])]
        writer.add_embedding(model.embeddings[geo].weight.detach().cpu(), metadata=labels,
                             tag=f"embedding/{pre.cat_cols[geo]}")
    writer.add_hparams({"lr": a.lr, "dropout": a.dropout, "weight_decay": a.weight_decay,
                        "lambda_reg": a.lambda_reg, "batch_size": a.batch_size}, best_metrics)
    writer.close()
    print(f"Logged to runs/{a.run_name}  ->  tensorboard --logdir runs")


if __name__ == "__main__":
    main()
