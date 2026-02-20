import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------
# 1) INPUT PKL PATHS
# -------------------------
detector_name_list = [
    '/kaggle/tmp/tsne_pkls/hybrid/tsne_dict_gend_effort_FaceForensics++.pkl',  # SEEN
    '/kaggle/tmp/tsne_pkls/hybrid/tsne_dict_gend_effort_Celeb-DF-v2.pkl',      # UNSEEN
]

# -------------------------
# 2) LOAD PKL
# -------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    feat = np.asarray(d["feat"]).reshape(len(d["feat"]), -1)
    label = np.asarray(d["label"]).astype(int)  # 0 real, 1 fake
    return feat, label

feat_seen, label_seen = load_pkl(detector_name_list[0])
feat_unseen, label_unseen = load_pkl(detector_name_list[1])

# Optional normalize (giữ như code gốc)
def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

feat_seen = l2_normalize(feat_seen)
feat_unseen = l2_normalize(feat_unseen)

# -------------------------
# 3) CONCAT
# -------------------------
feat = np.concatenate([feat_seen, feat_unseen], axis=0)

rf_label = np.concatenate([label_seen, label_unseen], axis=0).astype(int)
seen_flag = np.concatenate([
    np.ones(len(label_seen), dtype=int),
    np.zeros(len(label_unseen), dtype=int)
])

# -------------------------
# 4) BUILD 4 GROUPS
# -------------------------
group = np.zeros_like(rf_label, dtype=int)
group[(seen_flag == 1) & (rf_label == 0)] = 0  # Seen Real
group[(seen_flag == 1) & (rf_label == 1)] = 1  # Seen Fake
group[(seen_flag == 0) & (rf_label == 0)] = 2  # Unseen Real
group[(seen_flag == 0) & (rf_label == 1)] = 3  # Unseen Fake

names   = ["Seen Real", "Seen Fake", "Unseen Real", "Unseen Fake"]
colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
markers = ["*", "o", "o", "*"]

# -------------------------
# 5) TSNE (GIỮ NGUYÊN CODE GỐC)
# -------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=250,
    random_state=1024
)

x2d = tsne.fit_transform(feat)

# -------------------------
# 6) PLOT (đẹp hơn nhưng không đổi layout)
# -------------------------
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# vẽ theo layer để nhìn rõ hơn
plot_order = [2, 3, 0, 1]  # Unseen Real, Unseen Fake, Seen Real, Seen Fake

size_map  = {0: 26, 1: 26, 2: 20, 3: 20}
alpha_map = {0: 0.85, 1: 0.90, 2: 0.80, 3: 0.80}

for g in plot_order:
    idx = np.where(group == g)[0]
    ax.scatter(
        x2d[idx, 0],
        x2d[idx, 1],
        s=size_map[g],
        alpha=alpha_map[g],
        c=colors[g],
        marker=markers[g],
        linewidths=0,
        label=names[g]
    )

ax.legend(loc="upper right", fontsize=18, frameon=True, markerscale=1.2)
ax.axis("off")

plt.tight_layout(pad=0.6)
plt.savefig("tsne_seen_unseen_clean.png", dpi=300, bbox_inches="tight")
print("Saved: tsne_seen_unseen_clean.png")
