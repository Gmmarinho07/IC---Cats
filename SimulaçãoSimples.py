#Estudo de regressão supervisionada
#Entrada X : Temeperatura, Catalisador, tempo e etc
#Saída y: Conversão da reação (%) == y = f(x)
#Estrutura de rede neural Feedforward == Input -> Camada Oculta -> Output
#Cada neurônico faz z = w.x+b, a = ativação(z)
# Objetivo: Minimizar o erro entre previsão e valor real, usando o MAE(erro absoluto médio)
#Métrica principal: R² (coeficiente de determinação) == 1, 1 ->perfeito, 0 -> ruim, <0 pior que média
#Pre-processamento: One-hot enconding para variáveis categóricas,e normalização de escalas.
#Outliers: Valores extremos que confundem o modelo
#Ovefitting: Modelo 'decora' os dados de treinamento, solução -> divisão treino/teste(70/30), regularização e validação
#Otimização: Optuna, testa automaticamente várias combinações.


#Tudo que posso fazer é um modelo simplificado.

#Dataset simulado
import pandas as pd
import numpy as np
import tensorflow as tf


np.random.seed(42) #Para reprodutibilidade

data = pd.DataFrame({
    "temperatura": np.random.randint(200, 800, 200), #Variável numérica
    "tempo": np.random.randint(1, 10, 200), #Variável numérica
    "catalisador": np.random.choice(["Ni", "Pt", "Cu"], 200) # Variável categórica 
})

# função de conversão
data ["conversão"] = (
    0.05 * data["temperatura"] + #A temperatura tem um impacto positivo na conversão
    2 * data["tempo"] + # O tempo também tem um impacto positivo, mas menor que a temperatura
    np.where(data["catalisador"] == "Ni", 20, 10) + np.random.randn(200) * 5
)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# separar X e y

X = data.drop("conversão", axis=1) #X = data[["temperatura", "tempo", "catalisador"]]
y = data["conversão"] # y = data["conversão"]

# One-hot enconding para a variável categórica

X = pd.get_dummies(X, columns = ["catalisador"])

#normalização de escalas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size = 0.3, random_state = 42)



#modelo de rede neural (aqui tive que pesquisar, não tinha memorizado a sintaxe exata)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, activation = 'relu', input_shape = (X_train.shape[1], )),
    Dense(16, activation = 'relu'),
    Dense(1) # Camada de saída para regressão
])

model.compile(
    optimizer = 'adam',
    loss = 'mae'

)

#Treinamento do modelo, igual ao artigo, 100 épocas e validação de 20% dos dados de treinamento para monitorar o desempenho do modelo durante o treinamento.
history = model.fit(X_train, y_train,
                    epochs = 100,
                    validation_split = 0.2,
                    verbose = 1) # verbose = 1 para mostrar o progresso do treinamento

#Avaliação do modelo

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R²: ", r2)


#Previsão
novo = pd.DataFrame({
    "temperatura": [500],
    "tempo": [5],
    "catalisador": ["Ni"]
})

novo = pd.get_dummies(novo) #One-hot encoding para a nova entrada
novo = novo.reindex(columns = X.columns, fill_value = 0) #Garantir que as colunas estejam na mesma ordem e preencher com 0 as colunas ausentes
novo_scale = scaler.transform(novo) #Normalização da nova entrada usando o mesmo scaler do treinamento

pred = model.predict(novo_scale) #Previsão da conversão para a nova entrada
print("Conversão prevista: ", pred[0][0]) #A previsão é um array 2D, então acessamos o valor com [0][0] para obter o número da conversão prevista.
