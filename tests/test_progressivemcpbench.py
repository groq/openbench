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

def test_record_to_sample_empty_answer_raises():
    record = {
        "task_id": "task_3",
        "Question": "Bad",
        "answers": [],
        "category": "Bad",
    }
    with pytest.raises(ValueError, match="Empty answers list"):
        record_to_sample(record)

def test_get_dataset_mocked():
    mock_data = [
        {
            "task_id": "1",
            "Question": "Q1",
            "answers": ["A1"],
            "category": "Test",
            "file_name": "f1",
            "Annotator Metadata": {}
        }
    ]
    mock_json = json.dumps(mock_data)
    
    with patch("openbench.datasets.progressivemcpbench.DATA_FILE", Path("/mock/path.json")), \
         patch("builtins.open", mock_open(read_data=mock_json)), \
         patch("openbench.datasets.progressivemcpbench._ensure_local_json_dataset") as mock_ensure:
        
        dataset = get_dataset()
        assert len(dataset) == 1
        assert dataset[0].id == "1"
        mock_ensure.assert_called_once()
