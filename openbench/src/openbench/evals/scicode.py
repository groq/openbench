"""
SCICode implementation.

Code attribution:

This implementation is adapted from the following repository:
https://github.com/scicode-bench/SciCode

Implemented by Minyang Tian et al.

As of August 13, 2025, this implementation uses the validation split of the dataset, due to a bug with the test split in this implementation.
When 'test' is runnable, revert to 'test'.
"""

from inspect_ai import Task, task


from openbench.datasets.scicode import return_hf_dataset
from openbench.scorers.scicode import scicode_scorer
from openbench.solvers.scicode import scicode_solver


@task
def scicode(
    split: str = "test",  # TODO: when 'test' is runnable, revert to 'test'
    output_dir: str = "./tmp",
    with_background: bool = False,
    h5py_file: str | None = None,
    mode: str = "normal",
):
    print(
        "As of August 13, 2025, this implementation uses the validation split of the dataset, due to a bug with the test split in this implementation."
    )
    print("When 'test' is runnable, revert to 'test'.")

    return Task(
        dataset=return_hf_dataset(split),
        solver=scicode_solver(
            output_dir=output_dir,  # type: ignore
            with_background=with_background,  # type: ignore
            mode=mode,  # type: ignore
        ),
        scorer=scicode_scorer(
            output_dir=output_dir,  # type: ignore
            with_background=with_background,  # type: ignore
            h5py_file=h5py_file,  # type: ignore
        ),
    )
