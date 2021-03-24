import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor


# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("model/data.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

st.image('./boston.jpeg')
# título
st.markdown(
    """
    <style>
        .container{
           border: 1px solid black;
           border-radius: 15px;
           display: flex;
           align-items: center;
           justify-content: center;
        }
        .container h1{
            font-size: 30px;
            margin: 0px;
        }
    </style>

    <div class="container">
        <h1>Data App - Prevendo Valores de Imóveis</h1>
    </div>
    """,unsafe_allow_html=True
)

# subtítulo


# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# atributos para serem exibidos por padrão
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
st.dataframe(data[cols].head(10))


st.subheader("Distribuição de imóveis por média de preço")

# definindo a faixa de valores
faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), float(data.MEDV.max()), (0.0, 50.0))

# filtrando os dados
dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

# plot a distribuição dos dados
f = px.histogram(dados, x="MEDV", nbins=200, title="Distribuição de Preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)


st.sidebar.subheader("Filtro de Atributos")

# mapeando dados do usuário para cada atributo
crim = st.sidebar.slider("Taxa de Criminalidade",0,90 )
indus = st.sidebar.slider("Proporção de Hectares de Negócio",0,30)
chas = st.sidebar.radio("Faz limite com o rio?",("Sim","Não"))

# transformando o dado de entrada em valor binário
chas = 1 if chas == "Sim" else 0

nox = st.sidebar.slider("Concentração de óxido nítrico",0.0,1.0)

rm = st.sidebar.slider("Número de Quartos", 0,9)

ptratio = st.sidebar.slider("Índice de alunos para professores",0,22)

b = st.sidebar.slider("Proporção de pessoas com descendencia afro-americana",0,397)

lstat = st.sidebar.slider("Porcentagem de status baixo",0,38)

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    st.subheader("O valor previsto para o imóvel é:")
    result = "US $ "+str(round(result[0]*100,2))
    st.write(result)