from calculator import process_query

print("=" * 50)
print("🤖 Simple Calculator AI Agent")
print("=" * 50)

while True:
    query = input("\nAsk me something (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye! 👋")
        break

    answer = process_query(query)
    print(answer)