from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory  # ✅ corrected here

llm = OllamaLLM(model="llama3", base_url="http://host.docker.internal:11434")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

memory = ConversationBufferMemory(return_messages=True)

agent = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=lambda session_id: memory
)

while True:
    user_input = input("You: ")
    result = agent.invoke({"input": user_input}, config={"configurable": {"session_id": "duncan"}})
    print("Agent:", result.content)

