from inspect_ai.dataset import Sample
from typing import Any
from inspect_ai.dataset import hf_dataset
import os
from typing import Literal
from Crypto.Cipher import AES
from base64 import b64decode
import requests  # type: ignore

CURR_TASK = {}

llamastack_mappping = {
    "ek/llamastack/half-book": "ek_half_book",
    "ek/llamastack/full-book": "ek_full_book",
    "ke/llamastack/half-book": "ke_half_book",
    "ke/llamastack/full-book": "ke_full_book",
}


def get_llamastack_mtob_dataset(subtask, curr_task):
    global CURR_TASK
    CURR_TASK = curr_task
    dataset = hf_dataset(
        "llamastack/mtob",
        split="test",
        name=llamastack_mappping[subtask],
        sample_fields=record_to_sample_llamastack,
    )
    return dataset


def get_groq_mtob_dataset(curr_task):
    global CURR_TASK
    CURR_TASK = curr_task
    dataset = hf_dataset(
        "groq/mtob",
        split="test",
        sample_fields=record_to_sample_groq,
    )
    return dataset


def record_to_sample_llamastack(record: dict[str, Any]) -> Sample:
    global CURR_TASK
    return Sample(
        input=record["messages"][0]["content"],
        metadata={
            "id": record["id"],
            "translation_task": CURR_TASK["translation_task"],
            "provider": CURR_TASK["provider"],
            "knowledge_base_task": CURR_TASK["knowledge_base_task"],
        },
        target=record["expected_answer"],
    )


def decrypt_text_aes_ctr(nonce, ciphertext):
    nonce = b64decode(nonce)
    ct = b64decode(ciphertext)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    pt = cipher.decrypt(ct)
    return pt.decode("utf-8")


def record_to_sample_groq(record: dict[str, Any]):
    global CURR_TASK

    translation_direction_dict = {
        "English_to_Kalamang": "ek",
        "Kalamang_to_English": "ke",
    }

    if translation_direction_dict[record["subtask"]] == CURR_TASK["translation_task"]:
        return Sample(
            input=decrypt_text_aes_ctr(
                nonce=record["original_nonce"],
                ciphertext=record["original_ciphertext"],
            ),
            metadata={
                "subtask": record["subtask"],
                "id": record["original_id"],
                "translation_direction": translation_direction_dict[record["subtask"]],
                "translation_task": CURR_TASK["translation_task"],
                "provider": CURR_TASK["provider"],
                "knowledge_base_task": CURR_TASK["knowledge_base_task"],
            },
            target=decrypt_text_aes_ctr(
                nonce=record["ground_truth_nonce"],
                ciphertext=record["ground_truth_ciphertext"],
            ),
        )
    else:
        return []


if os.getenv("MTOB_KEY") is None:
    raise ValueError(
        "MTOB_KEY is not set. Please view the README.md for MTOB for more details."
    )
key = os.getenv("MTOB_KEY").encode()  # type: ignore


def get_claude_book_medium() -> str:
    return ""


def get_claude_book_long() -> str:
    return ""


def get_context_for_groq_ds(
    context_type: Literal["claude-book-medium", "claude-book-long", "zero-shot"],
) -> str:
    if context_type == "claude-book-medium":
        resp = requests.get(
            "https://huggingface.co/datasets/Groq/mtob/raw/main/reference/grammar_book_for_claude_medium_encrypted_ct.txt"
        )
        medium_book_ct = resp.text
        resp = requests.get(
            "https://huggingface.co/datasets/Groq/mtob/raw/main/reference/grammar_book_for_claude_medium_encrypted_nonce.txt"
        )
        medium_book_nonce = resp.text

        medium_book_decrypted = decrypt_text_aes_ctr(
            nonce=medium_book_nonce, ciphertext=medium_book_ct
        )
        return_context = f"""
To help with the translation, here is the full text of a Kalamang-English grammar
book:
—
{medium_book_decrypted}
This is the end of the Kalamang-English grammar book.
—"""

        return return_context

    elif context_type == "claude-book-long":
        resp = requests.get(
            "https://huggingface.co/datasets/Groq/mtob/raw/main/reference/grammar_book_for_claude_long_encrypted_ct.txt"
        )
        long_book_ct = resp.text
        resp = requests.get(
            "https://huggingface.co/datasets/Groq/mtob/raw/main/reference/grammar_book_for_claude_long_encrypted_nonce.txt"
        )
        long_book_nonce = resp.text
        long_book_decrypted = decrypt_text_aes_ctr(
            nonce=long_book_nonce, ciphertext=long_book_ct
        )

        return_context = f"""
To help with the translation, here is the full text of a Kalamang-English grammar
book:
—
{long_book_decrypted}
This is the end of the Kalamang-English grammar book.
—"""

        return return_context

    elif context_type == "zero-shot":
        return ""
    else:
        raise ValueError(
            f"Invalid context type: {context_type}. Must be either 'claude-book-medium', 'claude-book-long', or 'zero-shot'."
        )


def create_mtob_user_prompt_for_groq_ds(
    source: Any,
    translation_direction: Literal["ek", "ke"],
    context_type: Literal["claude-book-medium", "claude-book-long", "zero-shot"],
) -> str:
    context = get_context_for_groq_ds(context_type)
    if translation_direction == "ek":
        USER_PROMPT = f"""
Kalamang is a language spoken on the Karas Islands in West Papua. Translate the
following sentence from English to Kalamang: {source}
{context}
Now write the translation. If you are not sure what the translation should be,
then give your best guess. Do not say that you do not speak Kalamang. If your
translation is wrong, that is fine, but provide a translation.
English: {source}
Kalamang translation:

Provide the translation in the following format:
Kalamang translation: <translation>
"""
        return USER_PROMPT
    elif translation_direction == "ke":
        USER_PROMPT = f"""
Kalamang is a language spoken on the Karas Islands in West Papua. Translate the
following sentence from Kalamang to English: {source}
{context}
Now write the translation. If you are not sure what the translation should be,
then give your best guess. Do not say that you do not speak Kalamang. If your
translation is wrong, that is fine, but provide a translation.
Kalamang: {source}
English translation:

Provide the translation in the following format:
English translation: <translation>
"""
        return USER_PROMPT
    else:
        raise ValueError(
            f"Invalid translation direction: {translation_direction}. Must be either 'ek' or 'ke'."
        )
