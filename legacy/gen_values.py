import pandas as pd

# Your current table (counts + percentages)
rows = [
    ("Benign", 56592, 36.93),
    ("CF",     17169, 11.21),
    ("CS",     15307,  9.99),
    ("CSD",    15421, 10.06),
    ("ISP",    15268,  9.96),
    ("CMP",    33466, 21.84),
]

df = pd.DataFrame(rows, columns=["Condition", "Count", "Percent"])

# --- Split CMP into 4 equal quarters and add to CF/CS/CSD/ISP ---
targets = ["CF", "CS", "CSD", "ISP"]

cmp_count = int(df.loc[df["Condition"] == "CMP", "Count"].iloc[0])
cmp_pct   = float(df.loc[df["Condition"] == "CMP", "Percent"].iloc[0])

q_count = cmp_count // 4
rem = cmp_count % 4  # distribute remainder so total is preserved

# counts to add to each target (first 'rem' get +1)
adds = {t: q_count + (1 if i < rem else 0) for i, t in enumerate(targets)}

# update counts
for t in targets:
    df.loc[df["Condition"] == t, "Count"] += adds[t]

# remove CMP row
df = df[df["Condition"] != "CMP"].reset_index(drop=True)

# recompute percentages from counts to keep consistency
total = df["Count"].sum()
df["Percent"] = df["Count"] / total * 100

# Optional: pretty formatting with commas + 2 decimals
df_display = df.copy()
df_display["Count"] = df_display["Count"].map(lambda x: f"{x:,}")
df_display["Percent"] = df_display["Percent"].map(lambda x: f"{x:.2f}")

print(df_display)

# If you want LaTeX rows:
print("\nLaTeX rows:")
for _, r in df_display.iterrows():
    print(f'{r["Condition"]} & {r["Count"]} & {r["Percent"]} \\\\')
