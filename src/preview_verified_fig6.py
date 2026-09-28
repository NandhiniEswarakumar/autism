import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    "results/comparison_boosting_70_features.csv",
    "results/comparison_neural_70_features.csv",
    "results/comparison_tabm_70_features.csv",
    "results/tabm_ple_best_f1.csv",
]

frames = [pd.read_csv(path).query("Threshold == 'Validation-best-F1'") for path in files]
table = pd.concat(frames, ignore_index=True)
table["Model"] = table["Model"].replace({"TabM": "TabM (base)", "TabM-PLE": "TabM-PLE"})
table = table.set_index("Model").loc[
    ["CatBoost", "LightGBM", "XGBoost", "MLP", "TabM-PLE", "TabNet", "TabM (base)"],
]

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
values = table[metrics].to_numpy()
print("Exact plotted values (percent):")
for model, row in zip(table.index, values):
    print(model + ": " + ", ".join(f"{metric}={value * 100:.2f}" for metric, value in zip(metrics, row)))

fig, ax = plt.subplots(figsize=(15, 6))
x = range(len(metrics))
width = 0.13
palette = plt.cm.tab10(__import__("numpy").linspace(0, 1, len(table)))
for index, (model, row) in enumerate(zip(table.index, values)):
    ax.bar([position + index * width for position in x], row,
           width=width, label=model,
           color=palette[index], edgecolor="white", linewidth=0.8)
ax.set_xticks([position + width * (len(table) - 1) / 2 for position in x])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim([0, 1.15])
ax.set_title("Model Comparison -- ASD Detection", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("results/plots/fig6_verified_70_models.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved results/plots/fig6_verified_70_models.png")