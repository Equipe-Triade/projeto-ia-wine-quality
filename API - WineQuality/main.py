from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

SEED = 21


app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
    return "Minha primeira API."

@app.route("/quality-wine/", methods=["POST"])
def algoritmoQualidadeVinho():
    if request.files:
        print("Arquivo recebido..")

        
        arquivo_upload = request.files["arquivo"]
        arquivo_upload.save(os.path.join(app.config["FILES_UPLOADED"], arquivo_upload.filename))

        print(f"Nome do arquivo: {arquivo_upload.filename}\n")
        caminho_arquivo = os.path.join(app.config["FILES_UPLOADED"], arquivo_upload.filename)

        df_vinho = pd.read_csv(caminho_arquivo, sep=";")


#         with open(caminho_arquivo, 'r') as arquivo:
#             csvreader = csv.reader(arquivo)

#             contador = 0
#             dados = []
#             for linha in csvreader:
#                 linha_atualizada = row[0].replace('"', '').replace("'", "").split(";")
#                 if contador == 0:
#                     nomes_colunas = linha_atualizada
#                     contador+=1

#                 else:
#                     dados.append(linha_atualizada)

        os.remove(caminho_arquivo)


        #df_vinho = pd.DataFrame(dados, columns=nomes_colunas)
        df_vinho = df_vinho.astype(float)
        df_vinho.dropna(inplace=True)

        df_vinho["class_quality"] = df_vinho["quality"].apply(lambda x: "alta" if x >= 7 else ("média" if x >= 5 else "baixa"))
        df_vinho.drop(columns="quality", inplace=True)


        y = df_vinho.pop("class_quality")
        X = df_vinho.copy()

        # Separando dados de treino, teste e validação
        X_treino_teste, X_validacao, y_treino_teste, y_validacao = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED, shuffle=True)


        # Normalizando os dados
        scaler = StandardScaler()

        scaler.fit(X_treino_teste)

        X_treino_teste_norm = scaler.transform(X_treino_teste)
        X_validacao_norm = scaler.transform(X_validacao)

        print("Construindo modelo...")
        # Antes de contruir o modelo, iremos tentar encontrar os melhores valores para alguns parâmetros, utilizando o GridSearchCV, que possui a validação cruzada
        parametros = {"max_depth": range(2, 100),
                    "min_samples_split": range(2, 100),
                    "min_samples_leaf": range(2, 100),
                    "criterion": ["gini", "entropy"]}

        arvore = DecisionTreeClassifier(random_state=SEED)

        busca_modelo = RandomizedSearchCV(arvore, parametros, n_iter=50, cv=StratifiedKFold(n_splits = 10, shuffle=True, random_state=SEED))
        busca_modelo.fit(X_treino_teste_norm, y_treino_teste)
        
        print("Treinando modelo e testando...")
        
        # Construção e treino do modelo
        modelo = busca_modelo.best_estimator_
        modelo.fit(X_treino_teste_norm, y_treino_teste)

        # Realizando previsão com os dados de validação
        previsao = modelo.predict(X_validacao_norm)

        print("Gerando resultados...\n")
        # Plotando a matriz de confusão
        cm = confusion_matrix(y_validacao, previsao, labels=modelo.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
        disp.plot(cmap="rocket")
        plt.title("Matriz de Confusão")
        plt.savefig("matriz_confusao.png")

        # Verificando métricas do modelo
        print(classification_report(y_validacao, previsao, zero_division=0))

        # Criando dicionário com os valores da matriz de confusão
        classes = {}
        valores = {}

        for i in range(0, len(modelo.classes_)):
            classes[f"{i}"] = modelo.classes_[i]
            for j in range(0, len(modelo.classes_)):
                valores[f"linha{i}xcoluna{j}"] = int(cm[i][j])


        dic_matriz_confusao = {
            "nomes_classes": classes,
            "valores": valores
        }


        # Criando json para retornar como resposta da requisição
        resultado = jsonify(acuracia_modelo=float(accuracy_score(y_validacao, previsao)), matriz_confusao=dic_matriz_confusao, metricas_de_cada_classe=classification_report(y_validacao, previsao, output_dict=True, zero_division=0), melhores_parametros_encontrados=busca_modelo.best_params_)
        print(resultado)
        print("\nResultados enviados!")
        return resultado

    else:
        return "Nenhum arquivo foi encontrado"


app.config["FILES_UPLOADED"] = "C:\\Users\\sosop\\OneDrive\\Documentos\\GitHub\\projeto-ia-wine-quality\\files_uploaded"

app.run(debug=True)
