from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=200)

while True:
    user =input("YOU: ")
    if user == "0":
        break
    response = model.invoke(user)
    print("BOT:",response.content)
    


