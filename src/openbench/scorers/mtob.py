from inspect_ai.scorer import scorer, Score, Scorer, accuracy, stderr, Target
from inspect_ai.solver import TaskState
import nltk  # type: ignore


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        score = nltk.translate.chrf_score.sentence_chrf(
            state.messages[-1].content.split(),  # type: ignore
            target.target[0].split(),  # type: ignore
        )
        return Score(value=score)

    return score
