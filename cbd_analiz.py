import glob
import gzip
import os
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats

# 1. AŞAMA: gse328789_extracted klasöründeki ham dosyalardan temiz matris oluşturma
print(
    "1. 'gse328789_extracted' içindeki ham dosyalardan CBD matrisi"
    " oluşturuluyor..."
)
extracted_files = sorted(
    glob.glob(os.path.join("gse328789_extracted", "*.tsv.gz"))
)
if not extracted_files:
  extracted_files = sorted(
      glob.glob(os.path.join("gse328789_extracted", "*.txt"))
  )

dfs = []
for f in extracted_files:
  sample_name = os.path.basename(f).split(".")[0]
  opener = gzip.open if f.endswith(".gz") else open
  with opener(f, "rt", encoding="utf-8", errors="ignore") as file:
    temp_df = pd.read_csv(file, sep=None, engine="python")
    if temp_df.shape[1] >= 2:
      temp_df = temp_df.iloc[:, [0, 1]]
      temp_df.columns = ["Gene", sample_name]
      dfs.append(temp_df)

if dfs:
  df_cbd = dfs[0]
  for d in dfs[1:]:
    df_cbd = pd.merge(df_cbd, d, on="Gene", how="outer")
  # Matrisi 'cbd_ekspr' olarak kaydedelim
  df_cbd.to_csv("cbd_ekspr", sep="\t", index=False)
  print(f"CBD Matrisi başarıyla oluşturuldu. Boyut: {df_cbd.shape}")
else:
  # Eğer matris daha önceden düzgün kaydedildiyse doğrudan okuyalım
  df_cbd = pd.read_csv("cbd_ekspr", sep="\t", engine="python")

# Gen sütununu standartlaştıralım
gene_col = None
for col in df_cbd.columns:
  if col and any(
      k in str(col).lower() for k in ["gene", "symbol", "id", "ens", "name"]
  ):
    gene_col = col
    break
if not gene_col:
  gene_col = df_cbd.columns[0]

df_cbd = df_cbd.rename(columns={gene_col: "Gene"})

# Sayısal örnek sütunlarını belirleyelim
sample_cols = [c for c in df_cbd.columns if c != "Gene"]
for c in sample_cols:
  df_cbd[c] = pd.to_numeric(df_cbd[c], errors="coerce").fillna(0)

print(f"Toplam Örnek Sütun Sayısı: {len(sample_cols)}")

# Grupları yarı yarıya ayıralım (Kontrol ve CBD)
n_half = len(sample_cols) // 2
ctrl_cols = sample_cols[:n_half]
cbd_cols = sample_cols[n_half:]

print(f"Kontrol Sütunları ({len(ctrl_cols)} adet)")
print(f"CBD Sütunları ({len(cbd_cols)} adet)")

# 2. AŞAMA: Ortalama, Log2 FC ve T-testi Hesaplama
print(
    "\n2. Ortalama, Log2 Fold Change ve İstatistiksel T-testi hesaplanıyor..."
)
df_cbd["Control_Mean"] = df_cbd[ctrl_cols].mean(axis=1)
df_cbd["CBD_Mean"] = df_cbd[cbd_cols].mean(axis=1)
df_cbd["Log2_FC"] = df_cbd["CBD_Mean"] - df_cbd["Control_Mean"]

p_vals = []
for idx, row in df_cbd.iterrows():
  try:
    c_vals = row[ctrl_cols].values.astype(float)
    t_vals = row[cbd_cols].values.astype(float)
    _, p_val = stats.ttest_ind(t_vals, c_vals)
    if np.isnan(p_val):
      p_val = 1.0
  except:
    p_val = 1.0
  p_vals.append(p_val)

df_cbd["p_value"] = p_vals
df_cbd["Neg_Log10_p"] = -np.log10(
    df_cbd["p_value"].clip(lower=1e-300) + 1e-300
)

# 3. Anlamlılık Eşikleri (p < 0.05 ve |Log2_FC| > 0.5)
p_thresh = 0.05
fc_thresh = 0.5

df_cbd["Regulasyon"] = "Anlamsız (Not Significant)"
df_cbd.loc[
    (df_cbd["Log2_FC"] > fc_thresh) & (df_cbd["p_value"] < p_thresh),
    "Regulasyon",
] = "Artan (Up-regulated)"
df_cbd.loc[
    (df_cbd["Log2_FC"] < -fc_thresh) & (df_cbd["p_value"] < p_thresh),
    "Regulasyon",
] = "Azalan (Down-regulated)"

print("\nCBD Gen Regülasyon Dağılımı:")
print(df_cbd["Regulasyon"].value_counts())

# 3. AŞAMA: İnteraktif Volkan Grafiği Oluşturma
print("\n3. İnteraktif Volkan Grafiği oluşturuluyor ve kaydediliyor...")
fig = px.scatter(
    df_cbd,
    x="Log2_FC",
    y="Neg_Log10_p",
    color="Regulasyon",
    color_discrete_map={
        "Artan (Up-regulated)": "crimson",
        "Azalan (Down-regulated)": "dodgerblue",
        "Anlamsız (Not Significant)": "darkgray",
    },
    hover_data=["Gene", "p_value"],
    title="CBD - İnteraktif Volkan Grafiği",
    labels={
        "Log2_FC": "Log2 Fold Change (CBD / Kontrol)",
        "Neg_Log10_p": "-log10(p-value)",
    },
)

fig.add_hline(
    y=-np.log10(p_thresh),
    line_dash="dash",
    line_color="black",
    annotation_text="p = 0.05 Eşiği",
)
fig.add_vline(
    x=fc_thresh,
    line_dash="dot",
    line_color="gray",
    annotation_text="Kat Değişim Eşiği",
)
fig.add_vline(x=-fc_thresh, line_dash="dot", line_color="gray")

fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(
    template="plotly_white",
    title_font_size=16,
    legend_title_text="Regülasyon Durumu",
)

output_html = "cbd_interaktif_volkan_grafigi.html"
fig.write_html(output_html)
print(f"\n[BAŞARILI] İnteraktif grafik '{output_html}' olarak kaydedildi!")

# 4. AŞAMA: Anlamlı Genleri g:Profiler İçin Kaydetme (İlk 800 Sınırı)
sig_cbd = df_cbd[df_cbd["Regulasyon"] != "Anlamsız (Not Significant)"]
cbd_gen_listesi = (
    sig_cbd.sort_values(by="p_value")["Gene"].dropna().astype(str).tolist()
)

if len(cbd_gen_listesi) > 800:
  cbd_gen_listesi = cbd_gen_listesi[:800]

with open("cbd_anlamli_genler.txt", "w", encoding="utf-8") as f:
  for g in cbd_gen_listesi:
    f.write(f"{g.strip()}\n")

print(
    f"\n[BAŞARILI] CBD için {len(cbd_gen_listesi)} adet anlamlı gen saf"
    " metin olarak 'cbd_anlamli_genler.txt' dosyasına kaydedildi!"
)
