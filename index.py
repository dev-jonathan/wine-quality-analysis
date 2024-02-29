import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Função para carregar dados de um arquivo CSV
def carregar_dados(caminho_arquivo):
    """Carrega o conjunto de dados do arquivo CSV especificado."""
    return pd.read_csv(caminho_arquivo, sep=';')

# Função para separar as características (X) da variável alvo (y)
def separar_caracteristicas_alvo(df, coluna_alvo):
    """Separa as características (X) da variável alvo (y)."""
    X = df.drop(coluna_alvo, axis=1)
    y = df[coluna_alvo]
    return X, y

# Função para normalizar os dados
def normalizar_dados(X):
    """Normaliza os dados para o intervalo entre 0 e 1 usando MinMaxScaler."""
    normalizador = MinMaxScaler()
    X_normalizado = normalizador.fit_transform(X)
    return X_normalizado

# Função para binarizar os dados
def binarizar_dados(X, threshold=0.5):
    """Binariza os dados com base em um limite especificado."""
    binarizador = Binarizer(threshold=threshold)
    X_binarizado = binarizador.fit_transform(X)
    return X_binarizado

# Função para padronizar os dados usando StandardScaler
def padronizar_dados(X):
    """Padroniza os dados usando StandardScaler."""
    escalador = StandardScaler()
    return escalador.fit_transform(X)

# Função para plotar histogramas para as características padronizadas
def plotar_histogramas(X, colunas):
    """Plota histogramas para as características padronizadas."""
    plt.figure(figsize=(20, 15))
    for i, coluna in enumerate(colunas):
        plt.subplot(4, 4, i+1)
        plt.hist(X[:, i], bins=20, alpha=0.7)
        plt.title(coluna)
    plt.tight_layout()
    plt.suptitle('Histogramas das Características Padronizadas')
    plt.show()

# Função para plotar gráficos de densidade para as características padronizadas
def plotar_graficos_densidade(X, colunas):
    """Plota gráficos de densidade para as características padronizadas."""
    plt.figure(figsize=(20, 15))
    for i, coluna in enumerate(colunas):
        plt.subplot(4, 4, i+1)
        densidade = gaussian_kde(X[:, i])
        xs = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 200)
        plt.plot(xs, densidade(xs))
        plt.title(coluna)
    plt.tight_layout()
    plt.suptitle('Gráficos de Densidade das Características Padronizadas')
    plt.show()

def main():
    # Caminho do arquivo de dados e coluna alvo
    caminho_arquivo = 'winequality-red.csv'
    coluna_alvo = 'quality'
    # Carrega os dados
    df = carregar_dados(caminho_arquivo)
    # Separa as características e a variável alvo
    X, y = separar_caracteristicas_alvo(df, coluna_alvo)
    # Padroniza os dados
    X_padronizado = padronizar_dados(X)
    # Normaliza os dados (opcional, escolha entre padronizar ou normalizar baseado na necessidade)
    X_normalizado = normalizar_dados(X)
    # Binariza os dados (exemplo com um limiar arbitrário)
    X_binarizado = binarizar_dados(X, threshold=0.5)
    # Plota histogramas das características (escolha o conjunto de dados apropriado)
    plotar_histogramas(X_padronizado, df.drop(coluna_alvo, axis=1).columns)
    # Plota gráficos de densidade das características (escolha o conjunto de dados apropriado)
    plotar_graficos_densidade(X_padronizado, df.drop(coluna_alvo, axis=1).columns)

if __name__ == "__main__":
    main()
