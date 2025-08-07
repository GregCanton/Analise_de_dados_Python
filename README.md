# analise-avc-lgpd
Predicting stroke mortality using anonymized real-world clinical data. Full ML pipeline: feature engineering, stats, logistic regression, random forest, performance metrics and PDF reporting.
- Análise de Fatores Associados ao Óbito em Pacientes com AVC
Este projeto investiga fatores clínicos e de exames de imagem associados ao risco de óbito em pacientes com Acidente Vascular Cerebral (AVC).
Os dados utilizados são reais e anonimizados, em conformidade com a Lei Geral de Proteção de Dados (LGPD), garantindo privacidade e segurança da informação.

- Objetivos
Identificar variáveis mais relevantes para o prognóstico de pacientes com AVC.

Comparar o desempenho de diferentes modelos preditivos na classificação do desfecho.

Automatizar a geração de relatório com resultados estatísticos e gráficos.

- Etapas do Projeto
Pré-processamento e anonimização da base de dados (exposicao_avc_anonimizado.xlsx).

Engenharia de variáveis: criação de DELTA_NIHSS (diferença entre NIHSS de entrada e alta).

Estatística descritiva das variáveis clínicas e de exames de imagem.

Testes de Mann-Whitney para comparação entre grupos (óbito vs. não-óbito).

Modelagem preditiva:

Regressão Logística.

Random Forest.

Avaliação de desempenho: Acurácia, AUC, matriz de confusão e gráficos de importância.

Geração automática de relatório PDF com métricas, tabelas e visualizações.

- Principais Resultados
Acurácia (Regressão Logística): 98%

AUC: 0.9887

Variáveis mais relevantes:

NIHSS na alta

Idade

DELTA_NIHSS

- Estrutura do Repositório
bash
Copiar código
analise_avc_exposicao_LGPD.py   # Script principal de análise
exposicao_avc_anonimizado.xlsx  # Base de dados anonimizada
relatorio_avc_gerado_final.pdf  # Relatório final com resultados
requirements.txt                # Lista de bibliotecas necessárias
- Como Executar
Instale as dependências:

pip install -r requirements.txt
Execute o script principal:

python analise_avc_exposicao_LGPD.py
O relatório será gerado automaticamente como relatorio_avc_gerado_final.pdf na pasta definida no código.

- Conformidade LGPD
Este projeto não contém dados pessoais identificáveis.
Todos os registros foram anonimizados ou adaptados para uso acadêmico e de demonstração.

- Tecnologias Utilizadas
Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy, FPDF)

Modelos Preditivos: Regressão Logística, Random Forest

Estatística: Teste de Mann-Whitney, correlações

Visualização: Matriz de correlação, importância de variáveis

Relatórios: Geração automática em PDF
