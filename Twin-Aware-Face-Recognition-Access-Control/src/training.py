"""
train_model.py
─────────────────────────────────────────────────────────────────────
Loads all face images from the dataset, trains the face encodings
+ LDA twin separators, and saves everything to a single .pkl file
(Python's pickle — .pth is PyTorch specific, .pkl is correct here
since we are using face_recognition + sklearn, not PyTorch).

Run this ONCE whenever you add new people or new images:
    python train_model.py

Output:  face_model.pkl
─────────────────────────────────────────────────────────────────────
"""

import face_recognition
import numpy as np
import os
import cv2
import pickle
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH         = r"C:\Users\HARINI\Downloads\dataset (1)\dataset"
MODEL_SAVE_PATH      = "face_model.pkl"       # output file
TOLERANCE            = 0.45
TWIN_GAP_THRESHOLD   = 0.10
LDA_ENABLED          = True
MIN_TWIN_TRAINING_IMGS = 3
TWIN_DISTANCE_CUTOFF = 0.60                   # pairs closer than this = possible twins

# ============================================================
# STAGE 1: LOAD & ENCODE ALL FACES
# ============================================================

encodeListKnown   = []
nameListKnown     = []
person_encode_map = defaultdict(list)

print("=" * 55)
print("  STAGE 1 — Loading & encoding faces from dataset")
print("=" * 55)

for person_name in sorted(os.listdir(DATASET_PATH)):
    person_folder = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_folder):
        continue

    loaded = 0
    for file in sorted(os.listdir(person_folder)):
        img_path = os.path.join(person_folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            print(f"  ⚠  No face detected in {file}, skipping.")
            continue

        enc = encodings[0]
        encodeListKnown.append(enc)
        nameListKnown.append(person_name)
        person_encode_map[person_name].append(enc)
        loaded += 1

    print(f"  ✓  {person_name}: {loaded} image(s) encoded")

all_people = list(person_encode_map.keys())
print(f"\n  Total: {len(encodeListKnown)} encodings | {len(all_people)} people: {all_people}")


# ============================================================
# STAGE 2: AUTO-DETECT TWIN PAIRS
# ============================================================

print("\n" + "=" * 55)
print("  STAGE 2 — Detecting similar-looking (twin) pairs")
print("=" * 55)

pairwise   = {}
twin_pairs = set()

for i, p1 in enumerate(all_people):
    for j, p2 in enumerate(all_people):
        if j <= i:
            continue
        dists = [
            np.linalg.norm(np.array(e1) - np.array(e2))
            for e1 in person_encode_map[p1]
            for e2 in person_encode_map[p2]
        ]
        mean_dist = np.mean(dists)
        pairwise[(p1, p2)] = mean_dist

        if mean_dist < TWIN_DISTANCE_CUTOFF:
            twin_pairs.add((p1, p2))
            twin_pairs.add((p2, p1))

print("\n  Pairwise mean distances:")
for (p1, p2), dist in sorted(pairwise.items(), key=lambda x: x[1]):
    flag = "  ⚠ POSSIBLE TWINS" if (p1,p2) in twin_pairs else ""
    print(f"    {p1} ↔ {p2}: {dist:.4f}{flag}")


# ============================================================
# STAGE 3: TRAIN LDA FOR TWIN PAIRS
# ============================================================

print("\n" + "=" * 55)
print("  STAGE 3 — Training LDA twin separators")
print("=" * 55)

twin_lda_models = {}

def train_lda_for_pair(p1, p2):
    X, y = [], []
    for enc in person_encode_map[p1]:
        X.append(enc); y.append(p1)
    for enc in person_encode_map[p2]:
        X.append(enc); y.append(p2)

    if len(X) < 4:
        print(f"  ⚠  Not enough samples for {p1}/{p2}")
        return None

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    lda   = LinearDiscriminantAnalysis()
    lda.fit(X, y_enc)

    acc = lda.score(X, y_enc)
    print(f"  ✓  LDA [{p1} vs {p2}]: training accuracy = {acc*100:.1f}%")
    return {'lda': lda, 'le': le}

if LDA_ENABLED:
    processed_pairs = set()
    for (p1, p2) in twin_pairs:
        key = tuple(sorted([p1, p2]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)

        n1 = len(person_encode_map[p1])
        n2 = len(person_encode_map[p2])
        if n1 >= MIN_TWIN_TRAINING_IMGS and n2 >= MIN_TWIN_TRAINING_IMGS:
            model = train_lda_for_pair(*key)
            if model:
                twin_lda_models[key] = model
        else:
            print(f"  ⚠  Skipping LDA for {p1}/{p2} — need ≥{MIN_TWIN_TRAINING_IMGS} images each")
else:
    print("  LDA disabled in config.")


# ============================================================
# STAGE 4: VISUALISE TWIN SEPARATION (saves as PNG, no blocking)
# ============================================================

print("\n" + "=" * 55)
print("  STAGE 4 — Visualising twin separation (PCA plot)")
print("=" * 55)

processed_viz = set()
for (p1, p2) in twin_pairs:
    key = tuple(sorted([p1, p2]))
    if key in processed_viz:
        continue
    processed_viz.add(key)

    X, colors, labels = [], [], []
    for enc in person_encode_map[p1]:
        X.append(enc); colors.append('royalblue'); labels.append(p1)
    for enc in person_encode_map[p2]:
        X.append(enc); colors.append('crimson');   labels.append(p2)

    if len(X) < 4:
        continue

    X_2d = PCA(n_components=2).fit_transform(X)
    seen = set()
    plt.figure(figsize=(7, 5))
    for xi, col, lbl in zip(X_2d, colors, labels):
        plt.scatter(xi[0], xi[1], c=col, s=90,
                    label=lbl if lbl not in seen else "")
        seen.add(lbl)
    plt.title(f"Twin Pair: {p1} vs {p2} — PCA of 128-d encodings")
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()

    plot_file = f"twin_pair_{p1}_vs_{p2}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"  📊 Plot saved → {plot_file}")

    status = "✓ LDA active" if key in twin_lda_models else "⚠ No LDA model"
    print(f"     {p1} vs {p2}: {status}")

if not twin_pairs:
    print("  No twin pairs found — skipping visualisation.")


# ============================================================
# STAGE 5: SAVE EVERYTHING TO face_model.pkl
# ============================================================

print("\n" + "=" * 55)
print("  STAGE 5 — Saving trained model")
print("=" * 55)

model_bundle = {
    "encodeListKnown"  : encodeListKnown,    # list of 128-d face encodings
    "nameListKnown"    : nameListKnown,       # matching name for each encoding
    "person_encode_map": dict(person_encode_map),  # name → list of encodings
    "all_people"       : all_people,          # list of all person names
    "twin_pairs"       : twin_pairs,          # set of (p1,p2) twin tuples
    "twin_lda_models"  : twin_lda_models,     # dict of LDA models per twin pair
    "pairwise"         : pairwise,            # pairwise distances
    "config": {
        "TOLERANCE"          : TOLERANCE,
        "TWIN_GAP_THRESHOLD" : TWIN_GAP_THRESHOLD,
        "LDA_ENABLED"        : LDA_ENABLED,
        "TWIN_DISTANCE_CUTOFF": TWIN_DISTANCE_CUTOFF,
    }
}

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\n  ✅ Model saved → {MODEL_SAVE_PATH}")
print(f"     People  : {all_people}")
print(f"     Encodings: {len(encodeListKnown)}")
print(f"     Twin pairs: {len(twin_lda_models)} LDA model(s) trained")
print("\n  ✅ Training complete! Now run flask_server.py\n")
