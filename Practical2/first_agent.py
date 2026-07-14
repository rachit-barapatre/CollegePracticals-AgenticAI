from dotenv import load_dotenv
 
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_tokens=1000,timeout=None,max_retries=2)

print("---------------------WELCOME TO CHATBOT---------------------")
print("----------------------Type 0 to exit---------------------")

name = input("Enter your name: ")

prompt = f"""
You are a friendly AI assistant.

The user's name is {name}.

Greet the user warmly in 2-3 sentences.
"""
response = model.invoke(prompt)
    
print("BOT: ", response.content)

while True:
    
    prompt = input("YOU: ")
    if prompt == "0":
        break
    response = model.invoke(prompt)
    
    print("BOT: ", response.content)