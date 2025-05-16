from flock.core.interpreter.python_interpreter import PythonInterpreter
from flock.core.logging.trace_and_logged import traced_and_logged


@traced_and_logged
def code_evaluate_math(expression: str) -> float:
    try:
        result = PythonInterpreter(
            {},
            [
                "os",
                "math",
                "random",
                "datetime",
                "time",
                "string",
                "collections",
                "itertools",
                "functools",
                "typing",
                "enum",
                "json",
                "ast",
            ],
            verbose=True,
        ).execute(expression)
        return result
    except Exception:
        raise


@traced_and_logged
def code_code_eval(python_code: str) -> str:
    try:
        result = PythonInterpreter(
            {},
            [
                "os",
                "math",
                "random",
                "datetime",
                "time",
                "string",
                "collections",
                "itertools",
                "functools",
                "typing",
                "enum",
                "json",
                "ast",
            ],
            verbose=True,
        ).execute(python_code)
        return result
    except Exception:
        raise
