import json
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
from inspect_ai.dataset import Sample

from openbench.datasets.progressivemcpbench import record_to_sample, get_dataset, DATA_FILE

def test_record_to_sample():
    record = {
        "task_id": "task_1",
        "Question": "What is 2+2?",
        "answers": ["4", "four"],
        "category": "Math",
        "file_name": "math.txt",
        "Annotator Metadata": {"difficulty": "easy"}
    }
    sample = record_to_sample(record)
    assert sample.id == "task_1"
    assert sample.input == "What is 2+2?"
    assert sample.target == ["4", "four"]
    assert sample.metadata["category"] == "Math"
    assert sample.metadata["file_name"] == "math.txt"

def test_record_to_sample_single_answer():
    record = {
        "task_id": "task_2",
        "Question": "Hi",
        "answers": "Hello",
        "category": "Chat",
        "file_name": None,
        "Annotator Metadata": {}
    }
    sample = record_to_sample(record)
    assert sample.target == ["Hello"]

def test_record_to_sample_null_answer_returns_none():
    record = {
        "task_id": "task_3",
        "Question": "Skip me",
        "answers": None,
        "category": "Skip",
    }
    sample = record_to_sample(record)
    assert sample is None

def test_get_dataset_mocked():
    mock_data = [
        {
            "task_id": "1",
            "Question": "Q1",
            "answers": ["A1"],
            "category": "Test",
            "file_name": "f1",
            "Annotator Metadata": {}
        },
        {
            "task_id": "2",
            "Question": "Q2",
            "answers": None,
            "category": "Test",
            "file_name": "f2",
            "Annotator Metadata": {}
        }
    ]
    mock_json = json.dumps(mock_data)
    
    with patch("openbench.datasets.progressivemcpbench.DATA_FILE", Path("/mock/path.json")), \
         patch("pathlib.Path.open", mock_open(read_data=mock_json)):
        
        dataset = get_dataset()
        assert len(dataset) == 1
        assert dataset[0].id == "1"
