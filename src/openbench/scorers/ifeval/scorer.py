from inspect_ai.scorer import scorer, Score, Scorer, accuracy, stderr, Target


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer(scorer_type: str) -> Scorer:
    """Custom scorer for IFEval."""

    async def score(state, target: Target) -> Score:
        from openbench.scorers.ifeval.evaluation_lib import (
            test_instruction_following_strict,
            test_instruction_following_loose,
            read_single_prompt_dict,
        )

        inp = read_single_prompt_dict(state.metadata)

        prompt_to_response = {inp.prompt: state.messages[-1].content}

        if scorer_type == "strict":
            strict_result = test_instruction_following_strict(inp, prompt_to_response)
            score_value = 1.0 if strict_result.follow_all_instructions else 0.0
            state.metadata["scorer"] = "strict"
            state.metadata["explanation"] = str(strict_result.follow_instruction_list)
        elif scorer_type == "loose":
            loose_result = test_instruction_following_loose(inp, prompt_to_response)
            score_value = 1.0 if loose_result.follow_all_instructions else 0.0
            state.metadata["scorer"] = "loose"
            state.metadata["explanation"] = str(loose_result.follow_instruction_list)
        else:
            raise ValueError(f"Invalid scorer type: {scorer_type}")

        # TODO: the following code is commented out since the paper implies
        # that strict and loose scoring should be evaluated separately,
        # though keeping the code here for future refernece.

        # TODO: start of code
        # Calculate score using all() over both strict and loose results
        # score_value = (
        #     1.0
        #     if all(
        #         [
        #             strict_result.follow_all_instructions,
        #             loose_result.follow_all_instructions,
        #         ]
        #     )
        #     else 0.0
        # )
        # TODO: end of code

        return Score(
            value=score_value,
            explanation=f"using {state.metadata['scorer']} scorer result: {state.metadata['explanation']}",
        )

    return score
