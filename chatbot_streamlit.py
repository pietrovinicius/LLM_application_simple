#04/10/2024
#@PLima

# # chatbot_streamlit.py

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import datetime

def agora():
    agora = datetime.datetime.now()
    return agora.strftime("%Y-%m-%d %H:%M:%S")

# --- Configuração Inicial ---

# Carrega a chave de API do arquivo .env
load_dotenv()
print(f"{agora()} - Variáveis de ambiente carregadas do .env")

# Configuração da página do Streamlit
st.set_page_config(page_title="Chatbot com Memória", page_icon="🧠")
st.title("🤖 Chatbot com Memória")
st.caption("Desenvolvido com LangChain, Google Gemini e Streamlit")

# --- Lógica do LangChain com Memória ---

# 1. Instanciar o modelo LLM
print(f"{agora()} - Instanciando o modelo LLM")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# 2. Criar o template do prompt com memória
# O 'MessagesPlaceholder' é a peça chave aqui. Ele diz ao LangChain:
# "Insira o histórico da conversa aqui".
print(f"{agora()} - Criando o template do prompt com memória")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente prestativo. Responda todas as perguntas da melhor forma possível."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 3. Criar e inicializar a memória
# A memória precisa ser armazenada no 'session_state' do Streamlit
# para que não se perca a cada nova interação do usuário.
print(f"{agora()} - Criando e inicializando a memória")
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# 4. Criar a LLMChain
# Note que agora usamos LLMChain, que é projetada para funcionar com memória.
# A sintaxe de pipe (LCEL) também pode ser usada, mas LLMChain é mais explícita aqui.
print(f"{agora()} - Criando a LLMChain")
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=True # verbose=True nos mostra no console o que a chain está pensando
)

# --- Interface Gráfica com Streamlit ---

# Inicializa o histórico de mensagens no session_state se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []
    print(f"{agora()} - Inicializando o histórico de mensagens")


# Exibe as mensagens antigas no chat
print(f"{agora()} - Exibindo as mensagens antigas no chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
print(f"{agora()} - Capturando a entrada do usuário")
if user_input := st.chat_input("Qual a sua pergunta?"):
    # Adiciona a mensagem do usuário ao histórico e à tela
    st.session_state.messages.append({"role": "user", "content": user_input})
    print(f"{agora()} - Adicionando a mensagem do usuário ao histórico e à tela")
    with st.chat_message("user"):
        st.markdown(user_input)

    # Envia a entrada para a chain e obtém a resposta
    print(f"{agora()} - Enviando a entrada para a chain e obtendo a resposta")
    with st.spinner("Pensando..."):
        response = conversation_chain.invoke({"input": user_input})

    # Adiciona a resposta do assistente ao histórico e à tela
    st.session_state.messages.append({"role": "assistant", "content": response["text"]})
    print(f"{agora()} - Adicionando a resposta do assistente ao histórico e à tela")
    with st.chat_message("assistant"):
        st.markdown(response["text"])