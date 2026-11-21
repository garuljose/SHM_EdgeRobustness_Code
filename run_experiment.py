import os, time, json, itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd

from corrupt import CORRUPTION_FNS, linear_interpolate
from models import BaselineCNN, ReconGRU, CompactEncoder, CompactDecoder, CompactClassifier, count_params

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

WIN, STRIDE = 256, 128
RATES = [0.10, 0.20, 0.30, 0.40]
CTYPES = ["MCAR", "MAR", "MNAR"]


FREQ_BAND_HZ = 20.0  # keep 0-20 Hz, covering the structure's first four elastic
                      # modes (1.3-8.5 Hz nominal-to-damaged range); this mirrors
                      # standard ambient-vibration SHM practice, which band-limits
                      # to the known modal range rather than analyzing full-band
                      # noise, and improves classifier stability substantially


def to_spectrum_batch(X, fs=200.0):
    """(N, T, C) raw/corrupted time-domain window -> (N, F_band, C) log-magnitude
    spectrum restricted to 0-FREQ_BAND_HZ. All classification heads operate on
    this frequency-domain representation, consistent with the established
    vibration/modal-feature paradigm for damage-sensitive features; the
    reconstruction baseline (ReconGRU) is the only model that reconstructs a
    time-domain waveform, since that is the literal task of the imputation
    method it approximates."""
    T = X.shape[1]
    Xf = np.fft.rfft(X, axis=1)
    mag = np.log1p(np.abs(Xf)).astype(np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    n_keep = max(int(np.searchsorted(freqs, FREQ_BAND_HZ)), 4)
    return mag[:, :n_keep, :]

# ---------------------------------------------------------------- data prep
def window_runs(runs, labels, win=WIN, stride=STRIDE):
    X, y, run_id = [], [], []
    for r, (run, lab) in enumerate(zip(runs, labels)):
        T = run.shape[0]
        for start in range(0, T - win + 1, stride):
            X.append(run[start:start + win])
            y.append(lab)
            run_id.append(r)
    return np.stack(X).astype(np.float32), np.array(y), np.array(run_id)


def split_by_run(n_runs, seed=0, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_runs)
    n_train = int(train_frac * n_runs)
    n_val = int(val_frac * n_runs)
    return set(idx[:n_train]), set(idx[n_train:n_train + n_val]), set(idx[n_train + n_val:])


def normalize(X, mu, sd):
    return (X - mu) / sd


# ---------------------------------------------------------------- corruption helpers
def random_corrupt_batch(Xb, rng, rate_choices=RATES):
    """Apply a random corruption type+rate to each sample in a batch (used to
    train ReconGRU and the proposed model's fine-tuning stage on a mixture of
    corruption conditions rather than a single fixed one)."""
    out = np.empty_like(Xb)
    for i in range(Xb.shape[0]):
        ctype = rng.choice(CTYPES)
        rate = rng.choice(rate_choices)
        xc, mask = CORRUPTION_FNS[ctype](Xb[i], rate, rng)
        out[i] = xc
    return out


def mask_spans(Xb, rng, mask_frac=0.3, span=12):
    """Random span masking for self-supervised pretraining."""
    out = Xb.copy()
    T = Xb.shape[1]
    for i in range(Xb.shape[0]):
        n_mask = int(mask_frac * T / span)
        for _ in range(n_mask):
            start = rng.integers(0, max(T - span, 1))
            out[i, start:start + span, :] = 0.0
    return out


# ---------------------------------------------------------------- training loops
def train_baseline_cnn(Xtr, ytr, Xval, yval, epochs=35, lr=2e-3):
    """Xtr/Xval are raw time-domain (clean) windows; converted to spectrum here."""
    model = BaselineCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(to_spectrum_batch(Xtr)).permute(0, 2, 1)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    Xval_t = torch.tensor(to_spectrum_batch(Xval)).permute(0, 2, 1)
    yval_t = torch.tensor(yval, dtype=torch.long)
    best_state, best_acc = None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xval_t).argmax(1)
            acc = (pred == yval_t).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def train_recon_gru(Xtr, epochs=25, lr=1.5e-3, rng=None):
    model = ReconGRU().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    clean_t = torch.tensor(Xtr)  # (N, T, C)
    n = clean_t.shape[0]
    batch = 64
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            clean_b = clean_t[idx].numpy()
            corrupt_b = random_corrupt_batch(clean_b, rng)
            xb = torch.tensor(corrupt_b)
            yb = torch.tensor(clean_b)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
    return model


def pretrain_encoder(Xtr, epochs=25, lr=2e-3, rng=None, feat=16):
    """Self-supervised masked-spectrum reconstruction: mask random frequency
    bands of the clean spectrum and train the encoder+decoder to reconstruct
    the full spectrum (spectral analogue of masked-signal pretraining, in the
    same family as SpecAugment-style masking used in self-supervised audio/
    sequence models)."""
    enc = CompactEncoder(feat=feat)
    dec = CompactDecoder(feat=feat)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.MSELoss()
    Xtr_spec = to_spectrum_batch(Xtr)
    n = Xtr_spec.shape[0]
    batch = 64
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            clean_b = Xtr_spec[idx]
            masked_b = mask_spans(clean_b, rng, mask_frac=0.3, span=8)
            xb = torch.tensor(masked_b).permute(0, 2, 1)
            yb = torch.tensor(clean_b).permute(0, 2, 1)
            opt.zero_grad()
            z = enc(xb)
            out = dec(z)
            # crop/pad to match length (pooling can shift length by 1)
            L = min(out.shape[-1], yb.shape[-1])
            loss = lossf(out[..., :L], yb[..., :L])
            loss.backward()
            opt.step()
    return enc


def finetune_classifier(encoder, Xtr, ytr, Xval, yval, epochs=35, lr=3e-3, rng=None, feat=16, freeze_encoder=False):
    model = CompactClassifier(encoder, feat=feat)
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    batch = 64
    Xval_c = random_corrupt_batch(Xval, rng)
    Xval_t = torch.tensor(to_spectrum_batch(Xval_c)).permute(0, 2, 1)
    yval_t = torch.tensor(yval, dtype=torch.long)
    best_state, best_acc = None, -1
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            clean_b = Xtr[idx]
            corrupt_b = random_corrupt_batch(clean_b, rng)
            xb = torch.tensor(to_spectrum_batch(corrupt_b)).permute(0, 2, 1)
            yb = torch.tensor(ytr[idx], dtype=torch.long)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xval_t).argmax(1)
            acc = (pred == yval_t).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


# ---------------------------------------------------------------- evaluation
def eval_full_signal(model, X_time, y):
    """X_time: raw or reconstructed TIME-DOMAIN array (N,T,C); converted to
    spectrum here before classification, so every caller passes time-domain
    data consistently."""
    Xt = torch.tensor(to_spectrum_batch(X_time)).permute(0, 2, 1)
    with torch.no_grad():
        logits = model(Xt)
        prob = torch.softmax(logits, 1)[:, 1].numpy()
        pred = logits.argmax(1).numpy()
    return accuracy_score(y, pred), f1_score(y, pred), roc_auc_score(y, prob)


def eval_on_corruption(model_kind, model_extra, X, y, ctype, rate, rng, baseline_clf=None):
    Xc = np.empty_like(X)
    for i in range(X.shape[0]):
        xc, mask = CORRUPTION_FNS[ctype](X[i], rate, rng)
        if model_kind == "naive_impute":
            xc = linear_interpolate(xc, mask)
        Xc[i] = xc

    if model_kind in ("full_signal", "naive_impute"):
        return eval_full_signal(model_extra, Xc, y)

    if model_kind == "recon_gru":
        xt = torch.tensor(Xc)
        with torch.no_grad():
            recon = model_extra(xt).numpy()
        return eval_full_signal(baseline_clf, recon, y)

    if model_kind in ("proposed", "proposed_no_pretrain"):
        Xt = torch.tensor(to_spectrum_batch(Xc)).permute(0, 2, 1)
        with torch.no_grad():
            logits = model_extra(Xt)
            prob = torch.softmax(logits, 1)[:, 1].numpy()
            pred = logits.argmax(1).numpy()
        return accuracy_score(y, pred), f1_score(y, pred), roc_auc_score(y, prob)


# ---------------------------------------------------------------- resource profiling
def profile_model(model, sample_input, n_reps=200, quantize=False):
    import copy
    m = copy.deepcopy(model).eval()
    if quantize:
        m = torch.quantization.quantize_dynamic(m, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    tmp_path = f"{OUT}/_tmp_model.pt"
    torch.save(m.state_dict(), tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024.0
    os.remove(tmp_path)
    with torch.no_grad():
        for _ in range(10):
            m(sample_input)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            m(sample_input)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / n_reps * 1000.0
    return size_kb, latency_ms, m


# ================================================================== MAIN
def main():
    rng = np.random.default_rng(7)
    t_start = time.time()

    d = np.load("outputs/raw_runs.npz")
    runs, labels = d["runs"], d["labels"]
    n_runs = runs.shape[0]

    tr_idx, val_idx, te_idx = split_by_run(n_runs, seed=1)
    print(f"Run split: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

    def subset(idx_set):
        rr = [runs[i] for i in range(n_runs) if i in idx_set]
        ll = [labels[i] for i in range(n_runs) if i in idx_set]
        return window_runs(rr, ll)

    Xtr_raw, ytr, _ = subset(tr_idx)
    Xval_raw, yval, _ = subset(val_idx)
    Xte_raw, yte, _ = subset(te_idx)

    mu = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    sd = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = normalize(Xtr_raw, mu, sd)
    Xval = normalize(Xval_raw, mu, sd)
    Xte = normalize(Xte_raw, mu, sd)
    print(f"Windows: train={Xtr.shape[0]} val={Xval.shape[0]} test={Xte.shape[0]}")

    # ---- train models
    print("Training BaselineCNN (full-signal / naive-impute backbone)...")
    baseline_clf, val_acc_bcnn = train_baseline_cnn(Xtr, ytr, Xval, yval)
    print(f"  clean val acc = {val_acc_bcnn:.3f}, params={count_params(baseline_clf)}")

    print("Training ReconGRU baseline...")
    recon_gru = train_recon_gru(Xtr, rng=rng)
    print(f"  params={count_params(recon_gru)}")

    print("Pretraining CompactEncoder (self-supervised masked reconstruction)...")
    encoder = pretrain_encoder(Xtr, rng=rng)

    print("Fine-tuning proposed CompactClassifier on corrupted data...")
    proposed, val_acc_prop = finetune_classifier(encoder, Xtr, ytr, Xval, yval, rng=rng)
    print(f"  val acc (corrupted) = {val_acc_prop:.3f}, params={count_params(proposed)}")

    print("Training ablation: proposed model WITHOUT self-supervised pretraining...")
    encoder_scratch = CompactEncoder()
    proposed_noPT, val_acc_noPT = finetune_classifier(encoder_scratch, Xtr, ytr, Xval, yval, rng=rng)
    print(f"  val acc (corrupted, no pretrain) = {val_acc_noPT:.3f}")

    # ---- clean-data reference (rate = 0)
    rows = []
    acc, f1, auc = eval_full_signal(baseline_clf, Xte, yte)
    rows.append(dict(model="Full-signal baseline", ctype="clean", rate=0.0, accuracy=acc, f1=f1, auc=auc))
    acc, f1, auc = eval_full_signal(baseline_clf, Xte, yte)
    rows.append(dict(model="Naive-impute baseline", ctype="clean", rate=0.0, accuracy=acc, f1=f1, auc=auc))
    with torch.no_grad():
        recon_clean = recon_gru(torch.tensor(Xte)).numpy()
    acc, f1, auc = eval_full_signal(baseline_clf, recon_clean, yte)
    rows.append(dict(model="Recon-GRU baseline", ctype="clean", rate=0.0, accuracy=acc, f1=f1, auc=auc))
    acc, f1, auc = eval_on_corruption("proposed", proposed, Xte, yte, "MCAR", 1e-6, rng)
    rows.append(dict(model="Proposed (ours)", ctype="clean", rate=0.0, accuracy=acc, f1=f1, auc=auc))
    acc, f1, auc = eval_on_corruption("proposed_no_pretrain", proposed_noPT, Xte, yte, "MCAR", 1e-6, rng)
    rows.append(dict(model="Proposed - no pretrain (ablation)", ctype="clean", rate=0.0, accuracy=acc, f1=f1, auc=auc))

    # ---- main corruption grid
    model_specs = [
        ("Full-signal baseline", "full_signal", baseline_clf, None),
        ("Naive-impute baseline", "naive_impute", baseline_clf, None),
        ("Recon-GRU baseline", "recon_gru", recon_gru, baseline_clf),
        ("Proposed (ours)", "proposed", proposed, None),
        ("Proposed - no pretrain (ablation)", "proposed_no_pretrain", proposed_noPT, None),
    ]

    for ctype in CTYPES:
        for rate in RATES:
            for name, kind, extra, base_clf in model_specs:
                acc, f1, auc = eval_on_corruption(kind, extra, Xte, yte, ctype, rate, rng, baseline_clf=base_clf)
                rows.append(dict(model=name, ctype=ctype, rate=rate, accuracy=acc, f1=f1, auc=auc))
            print(f"  done {ctype} rate={rate}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/results_grid.csv", index=False)
    print("Saved results_grid.csv", df.shape)

    # ---- resource profiling (size / latency, pre- and post-quantization)
    sample = torch.tensor(to_spectrum_batch(Xte[:1])).permute(0, 2, 1)
    sample_gru = torch.tensor(Xte[:1])
    prof_rows = []
    for name, m, inp, quant_targets in [
        ("Full-signal baseline (CNN)", baseline_clf, sample, True),
        ("Recon-GRU baseline", recon_gru, sample_gru, True),
        ("Proposed (ours)", proposed, sample, True),
    ]:
        size_fp32, lat_fp32, _ = profile_model(m, inp, quantize=False)
        size_int8, lat_int8, _ = profile_model(m, inp, quantize=True)
        prof_rows.append(dict(model=name, size_kb_fp32=size_fp32, latency_ms_fp32=lat_fp32,
                               size_kb_int8=size_int8, latency_ms_int8=lat_int8,
                               params=count_params(m)))
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(f"{OUT}/resource_profile.csv", index=False)
    print(prof_df)

    # ---- quantized-model accuracy check (proposed model only, at rate=0.3 MCAR/MAR/MNAR)
    _, _, proposed_q = profile_model(proposed, sample, quantize=True)
    quant_rows = []
    for ctype in CTYPES:
        for rate in [0.2, 0.3]:
            Xc = np.empty_like(Xte)
            for i in range(Xte.shape[0]):
                xc, mask = CORRUPTION_FNS[ctype](Xte[i], rate, rng)
                Xc[i] = xc
            Xt = torch.tensor(to_spectrum_batch(Xc)).permute(0, 2, 1)
            with torch.no_grad():
                pred_fp32 = proposed(Xt).argmax(1).numpy()
                pred_int8 = proposed_q(Xt).argmax(1).numpy()
            quant_rows.append(dict(ctype=ctype, rate=rate,
                                    acc_fp32=accuracy_score(yte, pred_fp32),
                                    acc_int8=accuracy_score(yte, pred_int8)))
    quant_df = pd.DataFrame(quant_rows)
    quant_df.to_csv(f"{OUT}/quantization_accuracy.csv", index=False)
    print(quant_df)

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
