import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer
import matplotlib.pyplot as plt

# Carregar o conjunto de dados (substitua 'caminho_para_o_arquivo' pelo caminho real do arquivo)
df = pd.read_csv('winequality-red.csv', sep=';')

# Separar as características (X) da variável alvo 'quality' (y)
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Normalização
scaler_min_max = MinMaxScaler()
X_normalized = scaler_min_max.fit_transform(X)

# Padronização
scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)

# Binarização (ajustar o limiar conforme necessário)
binarizer = Binarizer(threshold=0.5)
X_binarized = binarizer.fit_transform(X)

# Visualização - Histogramas das características padronizadas
plt.figure(figsize=(20, 15))
for i in range(X_standardized.shape[1]):
    plt.subplot(5, 3, i+1)
    plt.hist(X_standardized[:, i], bins=20)
    plt.title(df.columns[i])
plt.tight_layout()
plt.show()
