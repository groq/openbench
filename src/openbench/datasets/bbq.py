from inspect_ai.dataset import hf_dataset, MemoryDataset, Sample


def load_dataset(subtask: str = "age"):
    if subtask == "age":
        dataset = hf_dataset(
            "Elfsong/BBQ", split=subtask, sample_fields=record_to_sample
        )

    else:
        list_of_rows = []
        list_of_splits = [
            "age",
            "disability_status",
            "gender_identity",
            "nationality",
            "physical_appearance",
            "race_ethnicity",
            "race_x_gender",
            "race_x_ses",
            "religion",
            "ses",
            "sexual_orientation",
        ]
        for split in list_of_splits:
            dataset = hf_dataset(
                "Elfsong/BBQ", split=split, sample_fields=record_to_sample
            )
            for row in dataset:
                list_of_rows.append(row)
        dataset = MemoryDataset(list_of_rows)

    return dataset


def record_to_sample(record):
    question = record["context"] + "\n\n" + record["question"]
    list_of_choices = [record["ans0"], record["ans1"], record["ans2"]]
    answer_label = record["answer_label"]
    int_to_char = {0: "A", 1: "B", 2: "C"}

    return Sample(
        input=question,
        choices=[
            str(list_of_choices[0]),
            str(list_of_choices[1]),
            str(list_of_choices[2]),
        ],
        target=int_to_char[answer_label],
    )
