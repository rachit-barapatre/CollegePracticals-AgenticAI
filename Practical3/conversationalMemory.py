from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=200
)

# Memory
chat_history = []

while True:
    user = input("YOU: ")

    if user == "0":
        break

    # Store user message
    chat_history.append(HumanMessage(content=user))

    # Send complete conversation to Gemini
    response = model.invoke(chat_history)

    # Print bot response
    print("BOT:", response.content)

    # Store bot reply
    chat_history.append(AIMessage(content=response.content))
    
    