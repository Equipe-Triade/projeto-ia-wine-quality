# projeto-ia-wine-quality

Passo a passo para a execução

Para o funcionamento correto do aplicativo, deve-se executar primeiro o arquivo para criação da API e depois rodar o aplicativo criado.
Abaixo seguem as instruções para tal execução:

1 - Entrar no prompt de comando e ir até a pasta do projeto com o caminho "..\projeto-ia-wine-quality\API - WineQuality"
2 - Executar o arquivo main.py com o comando:
>>python main.py
3 - Após execução finalizada, ir até a pasta do projeto com o caminho "..\projeto-ia-wine-quality\app-wine-quality"
4 - Para rodar o aplicativo, escrever o seguinte comando no cmd:
>>ionic serve
5 - Após execução finalizada, abrir o aplicativo mobile via browser


Para testar o funcionamento do aplicativo, deve-se submeter uma das base de dados em formato csv do link a seguir:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/


Além disso, para mais detalhes sobre essas bases, pode-se acessar o link a seguir:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality





OBS: Se for a primeira vez executando o app após dar um git clone, seguir passos a seguir antes dos citados acima:

1 - Após dar um git clone, abrir prompt de comando e ir até a pasta do projeto com o caminho "..\projeto-ia-wine-quality\app-wine-quality"
2 - Digitar o seguinte comando para instalar as dependências (lembrando de já ter o NodeJS instalado no computador):
>> npm install
3 - Ao terminar as instalações, abrir o arquivo main.py (que está dentro de "..\projeto-ia-wine-quality\API - WineQuality") em um editor de texto
4 - Ir até a linha 141 e adaptar o caminho de acordo com sua máquina local, como exemplificado a seguir:

app.config["FILES_UPLOADED"] = "C:\\Users\\nome_usuario\\Documentos\\projeto-ia-wine-quality\\files_uploaded"

5 - Após esses passos, é necessário fazer as instalações das bibliotecas python para que possa rodar a API (seria interessante fazer isso dentro de um ambiente virtual)
6 - Com isso, no cmd, ir até a pasta do projeto com o caminho "..\projeto-ia-wine-quality\API - WineQuality" e executar o seguinte comando:
>>pip install -r requirements.txt
7 - Finalizada as instalações das dependências, ir para o passo a passo principal (descrito no início)

