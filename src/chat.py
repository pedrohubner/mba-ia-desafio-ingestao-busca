from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}

def main():
    print("Chat iniciado. Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ").strip()

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando chat.")
            break

        response = search_prompt(question)
        print(f"\nAssistente: {response.content}\n")

if __name__ == "__main__":
    main()