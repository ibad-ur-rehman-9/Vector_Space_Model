from vsm_retrieval import search

def run_cli():
    print("=" * 50)
    print("  VSM Information Retrieval System")
    print("  23k0517 - Ibad Ur Rehman")
    print("=" * 50)

    while True:
        query = input("\Write your query or say 'exit' ").strip()

        if query.lower() == "exit":
            print("Bye!")
            break

        if not query:
            continue

        results = search(query)

        if not results:
            print("No doc found!")
            continue

        print(f"\n[+] {len(results)} documents found after filtering::\n")
        print(f"{'Rank':<6} {'Doc ID':<10} {'Score':<12}")
        print("-" * 30)

        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank:<6} {doc_id:<10} {score:.6f}")

if __name__ == "__main__":
    run_cli()