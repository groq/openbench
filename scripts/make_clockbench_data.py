"""Script to make the Clockbench dataset for upload to Hugging Face given input.json, answer.json, and image files."""

import json
import os
import pandas as pd


INPUT_JSON_FILE = ""  # path to the input json file
ANSWER_JSON_FILE = ""  # path to the answer json file
IMAGES_DIR = ""  # path to the images directory

with open(INPUT_JSON_FILE, "r") as f:
    input_data = json.load(f)
with open(ANSWER_JSON_FILE, "r") as f:
    answer_data = json.load(f)

# extract input data into dictionary
dataset = {}
for key, value in input_data.items():
    # check if all required keys exist in input data
    required_input_keys = [
        "image_url",
        "question_time",
        "question_shift",
        "question_angle",
        "question_zone",
    ]
    if not all(k in value for k in required_input_keys):
        print(
            f"Warning: Missing required keys for {key}. Required: {required_input_keys}"
        )
        continue

    # load image as bytes
    image_path = os.path.join(IMAGES_DIR, os.path.basename(value["image_url"]))
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        continue

    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    dataset[key] = {
        "id": key,
        "image": image_bytes,  # store bytes directly
        "file_name": value["image_url"],
        "question_time": value["question_time"],
        "question_shift": value["question_shift"],
        "question_angle": value["question_angle"],
        "question_zone": value["question_zone"],
        "target_time": {},
        "target_shift": {},
        "target_angle": {},
        "target_zone": {},
    }


# add answer data to dataset
for key, value in answer_data.items():
    # check if all required keys exist in answer data
    required_answer_keys = [
        "answer_time",
        "answer_shift",
        "answer_angle",
        "answer_zone",
    ]
    if not all(k in value for k in required_answer_keys):
        print(
            f"Warning: Missing required answer keys for {key}. Required: {required_answer_keys}"
        )
        continue

    # check if the key exists in the dataset
    if key not in dataset:
        print(f"Warning: No matching input data found for answer key {key}")
        continue

    # parse JSON strings to actual dictionaries and store in flattened structure
    for answer_type in ["time", "shift", "angle", "zone"]:
        answer_key = f"answer_{answer_type}"
        target_key = f"target_{answer_type}"
        try:
            dataset[key][target_key] = json.loads(value[answer_key])
        except json.JSONDecodeError:
            # string fallbacnk
            print(f"Warning: Invalid JSON in {answer_key} for {key}")
            dataset[key][target_key] = value[answer_key]

# convert dataset to pandas DataFrame and save as parquet
records_list = []
for record in dataset.values():
    # convert all target fields to JSON strings for consistent typing
    clean_record = record.copy()
    for key in ["target_time", "target_shift", "target_angle", "target_zone"]:
        if key in clean_record:
            clean_record[key] = json.dumps(clean_record[key])
    records_list.append(clean_record)

df = pd.DataFrame(records_list)

# save as parquet for upload to hf
output_file = "scripts/data.parquet"
df.to_parquet(output_file, index=False)

print(f"Created dataset with {len(dataset)} records in {output_file}")
print(f"Columns: {list(df.columns)}")
print(f"Parquet file size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
