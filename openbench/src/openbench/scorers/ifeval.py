"""Instruction Following scorer implementation."""

import json
import re
import functools
import langdetect  # type: ignore[import-untyped,import-not-found]
import nltk  # type: ignore[import-untyped,import-not-found]

from typing import Dict, Callable
from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    Scorer,
)
from inspect_ai.solver import TaskState
from openbench.metrics.ifeval import ifeval_metrics


class InstructionChecker:
    """Base class for instruction checkers."""

    def __init__(self, instruction_id: str):
        self.id = instruction_id

    def build_description(self, **kwargs):
        """Process and store instruction parameters."""
        # Default: store all kwargs as instance variables
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def check_following(self, response: str) -> bool:
        raise NotImplementedError


class CommaChecker(InstructionChecker):
    """Check for no commas."""

    def check_following(self, response: str) -> bool:
        return "," not in response


class LowercaseLettersEnglishChecker(InstructionChecker):
    """Check all lowercase English."""

    def check_following(self, response: str) -> bool:
        return check_english_with_case(response, str.islower)


class CapitalLettersEnglishChecker(InstructionChecker):
    """Check all capital letters English."""

    def check_following(self, response: str) -> bool:
        return check_english_with_case(response, str.isupper)


def check_english_with_case(
    response: str,
    case_predicate: Callable[[str], bool],
) -> bool:
    """
    Generic English text checker with casing rules.
    """
    if not response.strip() or not case_predicate(response):
        return False
    try:
        return langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return False


class PlaceholderChecker(InstructionChecker):
    """Check number of placeholders like [name]."""

    def build_description(self, *, num_placeholders=None, **kwargs):
        self._num_placeholders = num_placeholders or 1

    def check_following(self, response: str) -> bool:
        placeholders = re.findall(r"\[.*?\]", response)
        return len(placeholders) >= self._num_placeholders


class HighlightSectionChecker(InstructionChecker):
    """Check number of highlighted sections like *section*."""

    def build_description(self, *, num_highlights=None, **kwargs):
        self._num_highlights = num_highlights or 1

    def check_following(self, response: str) -> bool:
        # Find *text* and **text** patterns
        highlights = re.findall(r"\*[^\n\*]+\*", response)
        double_highlights = re.findall(r"\*\*[^\n\*]+\*\*", response)

        count = sum(1 for h in highlights if h.strip("*").strip())
        count += sum(
            1
            for h in double_highlights
            if h.removeprefix("**").removesuffix("**").strip()
        )

        return count >= self._num_highlights


class NumberOfWords(InstructionChecker):
    """Check word count constraints."""

    def build_description(self, *, relation=None, num_words=None, **kwargs):
        self._relation = relation or "at least"
        self._num_words = num_words or 100

    def check_following(self, response: str) -> bool:
        # Use proper tokenization like original
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(response)
        word_count = len(tokens)

        if self._relation == "at least":
            return word_count >= self._num_words
        elif self._relation == "less than":
            return word_count < self._num_words
        return False


class TitleChecker(InstructionChecker):
    """Check for title in double angular brackets <<title>>."""

    def check_following(self, response: str) -> bool:
        titles = re.findall(r"<<[^\n]+>>", response)
        return any(title.lstrip("<").rstrip(">").strip() for title in titles)


class RepeatPromptThenAnswer(InstructionChecker):
    """Check if response starts with repeated prompt."""

    def build_description(self, *, prompt_to_repeat=None, **kwargs):
        self._prompt_to_repeat = prompt_to_repeat or ""

    def check_following(self, response: str) -> bool:
        return self._prompt_to_repeat and response.strip().lower().startswith(
            self._prompt_to_repeat.strip().lower()
        )


class BulletListChecker(InstructionChecker):
    """Check exact number of bullet points."""

    def build_description(self, *, num_bullets=None, **kwargs):
        self._num_bullets = num_bullets or 1

    def check_following(self, response: str) -> bool:
        # Check both * and - style bullets
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
        return len(bullet_lists) + len(bullet_lists_2) == self._num_bullets


class SectionChecker(InstructionChecker):
    """Check for multiple sections with specific splitter."""

    def build_description(self, *, section_spliter=None, num_sections=None, **kwargs):
        self._section_splitter = section_spliter or "Section"
        self._num_sections = num_sections or 2

    def check_following(self, response: str) -> bool:
        pattern = r"\s?" + self._section_splitter + r"\s?\d+\s?"
        sections = re.split(pattern, response)
        return len(sections) - 1 >= self._num_sections


class CapitalWordFrequencyChecker(InstructionChecker):
    """Check frequency of all-caps words."""

    def build_description(
        self, *, capital_relation=None, capital_frequency=None, **kwargs
    ):
        self._relation = capital_relation or "at least"
        self._frequency = capital_frequency or 1

    def check_following(self, response: str) -> bool:
        capital_words = list(filter(str.isupper, nltk.word_tokenize(response)))
        count = len(capital_words)

        if self._relation == "at least":
            return count >= self._frequency
        elif self._relation == "less than":
            return count < self._frequency
        return False


class QuotationChecker(InstructionChecker):
    """Check if wrapped in double quotes."""

    def check_following(self, response: str) -> bool:
        response = response.strip()
        return len(response) > 1 and response[0] == '"' and response[-1] == '"'


class NumberOfSentences(InstructionChecker):
    """Check sentence count constraints."""

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _get_sentence_tokenizer():
        return nltk.data.load("nltk:tokenizers/punkt/english.pickle")

    def build_description(self, *, relation=None, num_sentences=None, **kwargs):
        self._relation = relation or "at least"
        self._num_sentences = num_sentences or 1

    def check_following(self, response: str) -> bool:
        # Use proper sentence tokenization like original
        tokenizer = self._get_sentence_tokenizer()
        sentences = tokenizer.tokenize(response)
        sentence_count = len(sentences)

        if self._relation == "at least":
            return sentence_count >= self._num_sentences
        elif self._relation == "less than":
            return sentence_count < self._num_sentences
        return False


class KeywordChecker(InstructionChecker):
    """Check for existence of specific keywords."""

    def build_description(self, *, keywords=None, **kwargs):
        self._keywords = keywords or []

    def check_following(self, response: str) -> bool:
        return all(
            re.search(r"\b" + re.escape(keyword) + r"\b", response, re.IGNORECASE)
            for keyword in self._keywords
        )


class ForbiddenWords(InstructionChecker):
    """Check that forbidden words are not present."""

    def build_description(self, *, forbidden_words=None, **kwargs):
        self._forbidden_words = forbidden_words or []

    def check_following(self, response: str) -> bool:
        return not any(
            re.search(r"\b" + re.escape(word) + r"\b", response, re.IGNORECASE)
            for word in self._forbidden_words
        )


class KeywordFrequencyChecker(InstructionChecker):
    """Check keyword frequency constraints."""

    def build_description(
        self, *, keyword=None, relation=None, frequency=None, **kwargs
    ):
        self._keyword = keyword or ""
        self._relation = relation or "at least"
        self._frequency = frequency or 1

    def check_following(self, response: str) -> bool:
        count = len(
            re.findall(
                r"\b" + re.escape(self._keyword) + r"\b", response, re.IGNORECASE
            )
        )

        if self._relation == "at least":
            return count >= self._frequency
        elif self._relation == "less than":
            return count < self._frequency
        return False


class LetterFrequencyChecker(InstructionChecker):
    """Check letter frequency constraints."""

    def build_description(
        self, *, letter=None, let_relation=None, let_frequency=None, **kwargs
    ):
        self._letter = (letter or "a").lower()
        self._relation = let_relation or "at least"
        self._frequency = let_frequency or 1

    def check_following(self, response: str) -> bool:
        count = response.lower().count(self._letter)

        if self._relation == "at least":
            return count >= self._frequency
        elif self._relation == "less than":
            return count < self._frequency
        return False


class ResponseLanguageChecker(InstructionChecker):
    """Check response language."""

    def build_description(self, *, language=None, **kwargs):
        self._language = language or "en"

    def check_following(self, response: str) -> bool:
        try:
            return langdetect.detect(response) == self._language
        except langdetect.LangDetectException:
            return True  # Count as instruction followed if detection fails


class ParagraphChecker(InstructionChecker):
    """Check paragraph count with *** separators."""

    def build_description(self, *, num_paragraphs=None, **kwargs):
        self._num_paragraphs = num_paragraphs or 1

    def check_following(self, value: str) -> bool:
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)
        empty_paragraphs = {
            i for i, paragraph in enumerate(paragraphs) if not paragraph.strip()
        }
        if empty_paragraphs - {0, num_paragraphs - 1}:
            return False
        return num_paragraphs - len(empty_paragraphs) == self._num_paragraphs


class PostscriptChecker(InstructionChecker):
    """Check for postscript with P.S. or P.P.S."""

    def build_description(self, *, postscript_marker=None, **kwargs):
        if postscript_marker == "P.P.S":
            self._pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif postscript_marker == "P.S." or not postscript_marker:
            self._pattern = r"\s*p\.\s?s\..*$"
        else:
            self._pattern = r"\s*" + postscript_marker + r".*$"

    def check_following(self, response: str) -> bool:
        return bool(re.search(self._pattern, response.lower()))


class EndChecker(InstructionChecker):
    """Check that response ends with specific phrase."""

    def build_description(self, *, end_phrase=None, **kwargs):
        self._end_phrase = end_phrase or ""

    def check_following(self, response: str) -> bool:
        if not self._end_phrase:
            return True
        return (
            response.strip()
            .strip('"')
            .lower()
            .endswith(self._end_phrase.strip().lower())
        )


class TwoResponsesChecker(InstructionChecker):
    """Check for two responses separated by ******."""

    def check_following(self, response: str) -> bool:
        responses = response.split("******")
        if {i for i, r in enumerate(responses) if not r.strip()} - {
            0,
            len(responses) - 1,
        }:
            return False
        valid_responses = list(filter(str.strip, responses))
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class JsonFormat(InstructionChecker):
    """Check if entire response is valid JSON."""

    def check_following(self, response: str) -> bool:
        # Strip markdown code blocks
        cleaned = response.strip()
        for prefix in ["```json", "```Json", "```JSON", "```"]:
            cleaned = cleaned.removeprefix(prefix)
        cleaned = cleaned.removesuffix("```").strip()

        try:
            json.loads(cleaned)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


class ConstrainedResponseChecker(InstructionChecker):
    """Check for constrained response options."""

    def check_following(self, response: str) -> bool:
        # Default options from original
        options = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
        response = response.strip()
        return any(option in response for option in options)


class ParagraphFirstWordCheck(InstructionChecker):
    """Check first word of nth paragraph."""

    def build_description(
        self, *, num_paragraphs=None, nth_paragraph=None, first_word=None, **kwargs
    ):
        self._num_paragraphs = num_paragraphs or 1
        self._nth_paragraph = nth_paragraph or 1
        self._first_word = (first_word or "").lower()

    def check_following(self, response: str) -> bool:
        # Split by double newlines
        paragraphs = re.split(r"\n\n", response)
        valid_paragraphs = list(filter(str.strip, paragraphs))

        if len(valid_paragraphs) != self._num_paragraphs or self._nth_paragraph > len(
            valid_paragraphs
        ):
            return False

        target_paragraph = valid_paragraphs[self._nth_paragraph - 1]
        if not target_paragraph.strip():
            return False
        words = target_paragraph.split()
        if not words:
            return False

        # Clean first word of punctuation
        actual_first_word = re.sub(r"[^\w]", "", words[0]).lower()
        return actual_first_word == self._first_word


INSTRUCTION_REGISTRY: Dict[str, type] = {
    "punctuation:no_comma": CommaChecker,
    "length_constraints:number_words": NumberOfWords,
    "length_constraints:number_sentences": NumberOfSentences,
    "keywords:forbidden_words": ForbiddenWords,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "keywords:frequency": KeywordFrequencyChecker,
    "startend:quotation": QuotationChecker,
    "combination:repeat_prompt": RepeatPromptThenAnswer,
    "keywords:existence": KeywordChecker,
    "change_case:english_lowercase": LowercaseLettersEnglishChecker,
    "detectable_format:title": TitleChecker,
    "keywords:letter_frequency": LetterFrequencyChecker,
    "language:response_language": ResponseLanguageChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "length_constraints:number_paragraphs": ParagraphChecker,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "startend:end_checker": EndChecker,
    "detectable_content:postscript": PostscriptChecker,
    "change_case:english_capital": CapitalLettersEnglishChecker,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "combination:two_responses": TwoResponsesChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:multiple_sections": SectionChecker,
    "length_constraints:nth_paragraph_first_word": ParagraphFirstWordCheck,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
}


@scorer(metrics=[ifeval_metrics()])
def ifeval_combined_scorer() -> Scorer:
    """Score instruction following compliance with both strict and loose evaluation.

    Returns combined metrics showing both strict and loose performance.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get instructions from metadata
        instruction_list = state.metadata.get("instruction_id_list", [])
        kwargs_list = state.metadata.get("kwargs", [])

        if not instruction_list:
            return Score(value=0.0, explanation="No instructions found")

        response = state.output.completion
        strict_following_list = []
        loose_following_list = []

        for idx, instruction_id in enumerate(instruction_list):
            instruction_cls = INSTRUCTION_REGISTRY[instruction_id]

            instruction = instruction_cls(instruction_id)
            instruction.build_description(**(kwargs_list[idx]))

            # Strict evaluation
            strict_followed = bool(response.strip()) and instruction.check_following(
                response
            )
            strict_following_list.append(strict_followed)

            # Loose evaluation
            responses = [
                response,
                response.replace("*", ""),
                "\n".join(response.split("\n")[1:]).strip(),  # Remove first line
                "\n".join(response.split("\n")[:-1]).strip(),  # Remove last line
                "\n".join(response.split("\n")[1:-1]).strip(),  # Remove both
            ]
            # Also try removing asterisks from trimmed versions
            responses.extend([r.replace("*", "") for r in responses[2:]])

            loose_followed = any(
                instruction.check_following(r) if r.strip() else False
                for r in responses
            )
            loose_following_list.append(loose_followed)

        explanations = [
            f"[S:{'✓' if strict_followed else '✗'} L:{'✓' if loose_followed else '✗'}] {instruction_id}"
            for instruction_id, strict_followed, loose_followed in zip(
                instruction_list,
                strict_following_list,
                loose_following_list,
                strict=True,
            )
        ]

        return Score(
            value=(1.0 if all(strict_following_list) else 0.0),
            answer=response,
            explanation="\n".join(explanations),
            metadata={
                "strict_follow_instruction_list": strict_following_list,
                "loose_follow_instruction_list": loose_following_list,
                "instruction_id_list": instruction_list,
            },
        )

    return score
