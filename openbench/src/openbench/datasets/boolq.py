from inspect_ai.dataset import Sample


def record_to_sample(record: dict) -> Sample:
    answer_label = record["answer"]
    int_to_char = {0: "A", 1: "B"}

    passage = record["passage"]
    question = record["question"]

    input_msg_pattern = f"{passage}\nQuestion: {question}?\nAnswer:"

    return Sample(
        input=input_msg_pattern,
        choices=["false", "true"],
        target=int_to_char[answer_label],
    )
