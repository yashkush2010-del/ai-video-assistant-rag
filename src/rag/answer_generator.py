def prepare_context(results):
    context = ""

    for result in results:
        context += (
            f"[{result['start']} - {result['end']}]\n"
            f"{result['text']}\n\n"
        )

    return context


if __name__ == "__main__":

    test_results = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "Hey buddy how was school?"
        },
        {
            "start": 4.0,
            "end": 8.0,
            "text": "It was mocking, Goop!"
        }
    ]

    context = prepare_context(test_results)

    print("Prepared context:")
    print(context)