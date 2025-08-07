
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from fpdf import FPDF
import os

# === Caminhos ===
arquivo_excel = r"c:\Users\gpcan\OneDrive\Área de Trabalho\Dados mestrado - Estudos\exposicao_avc_anonimizado.xlsx"
saida_pdf = r"c:\Users\gpcan\OneDrive\Área de Trabalho\Dados mestrado - Estudos\relatorio_avc_gerado_final.pdf"
pasta_graficos = r"c:\Users\gpcan\OneDrive\Área de Trabalho\Dados mestrado - Estudos\graficos_avc"
os.makedirs(pasta_graficos, exist_ok=True)

def limpar_unicode(texto):
    return str(texto).replace("–", "-").replace("→", "->").replace("•", "-").replace("’", "'")

# === 1. Leitura e preparação dos dados ===
df = pd.read_excel(arquivo_excel)
df["DELTA_NIHSS"] = df["NIHSS_ENTRADA"] - df["NIHSS_ALTA"]

# === 2. Estatísticas descritivas ===
descricoes = df.describe()

# === 3. Mann-Whitney ===
variaveis_continuas = ["DOSE EFETIVA (mSv)", "NIHSS_ENTRADA", "NIHSS_ALTA"]
grupo_morte = df[df["OBITO"] == 1]
grupo_vivo = df[df["OBITO"] == 0]
resultados_mwu = []

for var in variaveis_continuas:
    stat, p = mannwhitneyu(grupo_morte[var], grupo_vivo[var], alternative="two-sided")
    media_morte = grupo_morte[var].mean()
    media_vivo = grupo_vivo[var].mean()
    resultados_mwu.append(f"{var} -> Média (Óbito=1): {media_morte:.2f}, Média (Óbito=0): {media_vivo:.2f}, p-valor: {p:.4f}")

# === 4. Correlação ===
colunas_correlacao = ["IDADE", "DOSE EFETIVA (mSv)", "NIHSS_ENTRADA", "NIHSS_ALTA", "DELTA_NIHSS", "OBITO"]
correlacao = df[colunas_correlacao].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
correlacao_path = os.path.join(pasta_graficos, "matriz_correlacao.png")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig(correlacao_path)
plt.close()

# === 5. Modelagem ===
X = df[["IDADE", "DOSE EFETIVA (mSv)", "NIHSS_ENTRADA", "NIHSS_ALTA", "DELTA_NIHSS"]]
y = df["OBITO"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regressão Logística
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_scaled, y)
y_pred_log = modelo_log.predict(X_scaled)
auc_log = roc_auc_score(y, modelo_log.predict_proba(X_scaled)[:, 1])
matriz_log = confusion_matrix(y, y_pred_log)
report_log = classification_report(y, y_pred_log, output_dict=True)
coeficientes = modelo_log.coef_[0]
resultados_log = [f"{nome}: {coef:.4f}" for nome, coef in zip(X.columns, coeficientes)]

# Random Forest
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X, y)
importancias = modelo_rf.feature_importances_
resultados_rf = [f"{nome}: {importancia:.4f}" for nome, importancia in zip(X.columns, importancias)]

plt.figure(figsize=(8, 6))
sns.barplot(x=importancias, y=X.columns)
plt.title("Importância das Variáveis (Random Forest)")
plt.xlabel("Importância")
plt.tight_layout()
importancia_path = os.path.join(pasta_graficos, "importancia_variaveis.png")
plt.savefig(importancia_path)
plt.close()

# === 6. PDF Final ===
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Relatório Analítico - AVC e Exposição", ln=True)

# Resumo metodológico
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Resumo Metodológico", ln=True)
pdf.set_font("Arial", "", 10)
pdf.multi_cell(0, 8, limpar_unicode(
    "Este relatório apresenta uma análise dos dados clínicos anonimizada conforme a LGPD. Foram utilizados testes estatísticos "
    "e modelos preditivos para avaliar associação entre variáveis clínicas e óbito em pacientes com AVC."
))

# Estatísticas
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "1. Estatísticas Descritivas", ln=True)
pdf.set_font("Arial", "", 10)
for col in descricoes.columns:
    media = descricoes[col]["mean"]
    desvio = descricoes[col]["std"]
    pdf.cell(0, 8, f"{col}: Média={media:.2f}, Desvio={desvio:.2f}", ln=True)

# Mann-Whitney
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "2. Testes de Mann-Whitney", ln=True)
pdf.set_font("Arial", "", 10)
for linha in resultados_mwu:
    pdf.multi_cell(190, 8, limpar_unicode(linha))

# Regressão Logística
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "3. Regressão Logística", ln=True)
pdf.set_font("Arial", "", 10)
for linha in resultados_log:
    pdf.cell(0, 8, limpar_unicode(linha), ln=True)
pdf.cell(0, 8, f"AUC: {auc_log:.4f}", ln=True)
pdf.cell(0, 8, f"Acurácia: {report_log['accuracy']:.2f}", ln=True)

# Random Forest
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "4. Random Forest", ln=True)
pdf.set_font("Arial", "", 10)
for linha in resultados_rf:
    pdf.cell(0, 8, limpar_unicode(linha), ln=True)

# Gráficos
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "5. Gráficos", ln=True)
pdf.image(correlacao_path, w=170)
pdf.ln(5)
pdf.image(importancia_path, w=170)

# Exportar PDF
try:
    pdf.output(saida_pdf)
    print("Relatório final gerado com sucesso.")
except Exception as e:
    print("Erro ao gerar PDF:", e)
