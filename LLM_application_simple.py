#04/10/2024
#@PLima

#https://python.langchain.com/docs/tutorials/llm_chain/

#Crie um aplicativo LLM simples com modelos de bate-papo e modelos de prompt

#Neste guia rápido, mostraremos como criar um aplicativo LLM simples com o LangChain. 
#Este aplicativo traduzirá textos do inglês para outro idioma. 
#Trata-se de um aplicativo LLM relativamente simples: 
# consiste em apenas uma chamada de LLM e alguns prompts. 
# 
#Ainda assim, é uma ótima maneira de começar a usar o LangChain: 
#muitos recursos podem ser criados com apenas alguns prompts e uma chamada de LLM!

# --- BOAS PRÁTICAS: Gerenciamento de Chave de API ---
# Importamos a função para carregar as variáveis de ambiente do nosso arquivo .env
from dotenv import load_dotenv
load_dotenv()
print("1. Variáveis de ambiente carregadas do .env")

# --- LÓGICA DO LANGCHAIN ---

# Importando os componentes necessários
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Instanciando o modelo de LLM
# CORREÇÃO: Usamos um nome de modelo mais moderno e específico ("gemini-1.5-flash").
# Isso resolve o erro '404 Not Found' e também o aviso de 'deprecation',
# pois este modelo não precisa mais do parâmetro de compatibilidade.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
print("2. Modelo LLM (Google Gemini) instanciado.")

# Definindo o modelo de prompt (Prompt Template)
# Esta é a estrutura da nossa pergunta, com espaços para preencher.
prompt_template = ChatPromptTemplate.from_template(
    "Traduza o seguinte texto de {idioma_entrada} para {idioma_saida}: {texto}"
)
print("3. Template do prompt criado.")

# Instanciando o parser de saída
output_parser = StrOutputParser()
print("4. Parser de saída instanciado.")

# Criando a "Chain" (Corrente) usando LangChain Expression Language (LCEL)
# Conectamos as peças na ordem em que os dados devem fluir.
chain = prompt_template | model | output_parser
print("5. Chain criada com sucesso: prompt | model | parser.")

# Invocando (executando) a Chain
print("\n--- INVOCANDO A CHAIN ---")
dados_de_entrada = {
    "idioma_entrada": "inglês",
    "idioma_saida": "português",
    "texto": "hello my friendly!"
}
print(f"Dados de entrada: {dados_de_entrada}")

resultado = chain.invoke(dados_de_entrada)

print("\n--- RESULTADO ---")
print(resultado)