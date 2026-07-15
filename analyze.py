import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/raw_results.csv")
df = df[df["control"].isna()]
pivot = df.groupby(["model","depth"])["success"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
for model, grp in pivot.groupby("model"):
    ax.plot(grp["depth"]*100, grp["success"]*100, marker="o", linewidth=2, label=model)

ax.axvspan(40, 60, alpha=0.08, color="red", label="Predicted trough zone")
ax.set_xlabel("Bug depth in context window (%)")
ax.set_ylabel("Fix success rate (%)")
ax.set_xticks([0, 5, 25, 50, 75, 95, 100])
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig("results/positional_bias_curve.png", dpi=150)