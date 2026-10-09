"""Tasks and grading for RL. Exercism: load_exercise/Task; toy: known-answer rewards;
more runners and task sources (loopgym.py, splits.py) land here. grade() is one entry
point for every task source, so the trainer cannot tell them apart."""
from . import toy
from .exercism import RUNNERS, Task, grade as _exercism_grade, load_exercise, runner_names

__all__ = ["RUNNERS", "Task", "grade", "load_exercise", "runner_names", "toy"]


def grade(task, files, runner="local", timeout=30, step=0, index=0):
    """Grade one completion for task -> (reward, info). Test-based sources (exercism and,
    later, loopgym) run the task's hidden tests via RUNNERS (runner/timeout select which);
    toy tasks score in-process via their reward function, keyed by (step, index)."""
    if isinstance(task, toy.ToyTask):
        return toy.grade(task, files, step=step, index=index)
    return _exercism_grade(task, files, runner=runner, timeout=timeout)
