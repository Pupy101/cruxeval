# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import tqdm
from giga import GigaChat

from .constants import REPO_DIR

logger = logging.getLogger(__name__)
client = GigaChat()


def make_cot_output_prompt(s: tuple[str, str]) -> str:
    code, inputs = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({inputs}) == ??
[/PYTHON]
[THOUGHT]
"""


def make_direct_output_prompt(s: tuple[str, str]) -> str:
    code, inputs = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({inputs}) == ??
[/PYTHON]
[ANSWER]
"""


def make_direct_input_prompt(s: tuple[str, str]) -> str:
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""


def make_cot_input_prompt(s: tuple[str, str]) -> str:
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""


def extract_answer_direct_output(gen: str) -> str:
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()


def extract_answer_direct_input(gen: str) -> str:
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()


def extract_answer_cot_input(gen: str) -> str:
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[0]
        if "assert f" in gen:
            gen = "f" + gen.split("assert f")[1].strip()
        return gen.strip()
    return gen.split("\n")[-1].strip()


def extract_answer_cot_output(gen: str) -> str:
    if "[ANSWER]" in gen:
        gen = gen.split("[ANSWER]")[1].strip()
        if "==" in gen:
            gen = gen.split("==")[1]
        return gen.strip()
    return gen.split("\n")[-1].strip()


def call_api(system: str, prompt: str, temperature: float, n: int, model: str, max_tokens: int) -> list[str]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    outputs: list[str] = []
    for _ in range(n):
        while True:
            try:
                result = client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                break
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("Catch exception: ", exc_info=exc)
                time.sleep(1)
        outputs.append(result["choices"][0]["message"]["content"])
    return outputs


def prompt_giga_general(
    make_prompt_fn: Callable[[tuple[str, str]], str],
    i: int,
    cache: dict[str, list[str]],
    gpt_query: tuple[str, str],
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    pbar: tqdm.tqdm,
) -> tuple[int, tuple[str, list[str]]]:

    prompt = make_prompt_fn(gpt_query)

    cache_key = f"{prompt}_{model}" if temperature == 0 else f"{prompt}_{model}_{str(temperature)}"

    result: list[str]
    if cache_key not in cache or (cache_key in cache and n > len(cache[cache_key])):
        cache_result = []
        if cache_key in cache:
            n -= len(cache[cache_key])
            cache_result = cache[cache_key]
        system = "You are an expert at Python programming, code execution, test case generation, and fuzzing."
        result = call_api(
            system=system, prompt=prompt, temperature=temperature, n=n, model=model, max_tokens=max_tokens
        )
        cache[cache_key] = cache_result + result
    else:
        result = cache[cache_key]
    pbar.update(n=1)
    return i, (cache_key, result)


def batch_prompt(
    fn: Callable,
    extraction_fn: Callable,
    queries,
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    parallel: int,
):
    cache_dir = REPO_DIR / "cache.json"
    cache_dir_tmp = REPO_DIR / "cache.json.tmp"
    cache_dir_bak = REPO_DIR / "cache.json.bak"
    try:
        with open(cache_dir, "r") as fp:
            cache = json.load(fp)
    except Exception:  # pylint: disable=broad-exception-caught
        with open(cache_dir, "w") as fp:
            json.dump({}, fp)
        cache = {}

    # run the generations
    with ThreadPoolExecutor(max_workers=parallel) as executor, tqdm.tqdm(total=len(queries)) as pbar:
        futures = [
            executor.submit(fn, i, cache, query, temperature, n, model, max_tokens, pbar)
            for i, query in enumerate(queries)
        ]
        results_with_id = [future.result() for future in futures]
    results_with_id.sort()
    results = [i[1] for i in results_with_id]

    # update the cache
    for cache_key, r in results:
        cache[cache_key] = r
    with open(cache_dir_tmp, "w") as fp:
        json.dump(cache, fp)
    os.rename(cache_dir, cache_dir_bak)
    os.rename(cache_dir_tmp, cache_dir)
    os.remove(cache_dir_bak)

    # parse the output
    gens = [i[1] for i in results]
    return [[(extraction_fn(i), i) for i in r] for r in gens]


# direct output prompt
def prompt_direct_output(
    i: int,
    cache: dict,
    gpt_query: tuple[str, str],
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    pbar: tqdm.tqdm,
):
    return prompt_giga_general(
        make_direct_output_prompt,
        i=i,
        cache=cache,
        gpt_query=gpt_query,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        pbar=pbar,
    )


def batch_prompt_direct_output(
    queries: list[tuple[str, str]], temperature: float, n: int, model: str, max_tokens: int, parallel: int
):
    return batch_prompt(
        prompt_direct_output,
        extract_answer_direct_output,
        queries=queries,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        parallel=parallel,
    )


# cot output prompt
def prompt_cot_output(
    i: int,
    cache: dict,
    gpt_query: tuple[str, str],
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    pbar: tqdm.tqdm,
):
    return prompt_giga_general(
        make_cot_output_prompt,
        i=i,
        cache=cache,
        gpt_query=gpt_query,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        pbar=pbar,
    )


def batch_prompt_cot_output(
    queries: list[tuple[str, str]], temperature: float, n: int, model: str, max_tokens: int, parallel: int
):
    return batch_prompt(
        prompt_cot_output,
        extract_answer_cot_output,
        queries=queries,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        parallel=parallel,
    )


# direct input prompt
def prompt_direct_input(
    i: int,
    cache: dict,
    gpt_query: tuple[str, str],
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    pbar: tqdm.tqdm,
):
    return prompt_giga_general(
        make_direct_input_prompt,
        i=i,
        cache=cache,
        gpt_query=gpt_query,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        pbar=pbar,
    )


def batch_prompt_direct_input(
    queries: list[tuple[str, str]], temperature: float, n: int, model: str, max_tokens: int, parallel: int
):
    return batch_prompt(
        prompt_direct_input,
        extract_answer_direct_input,
        queries=queries,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        parallel=parallel,
    )


# cot input prompt
def prompt_cot_input(
    i: int,
    cache: dict,
    gpt_query: tuple[str, str],
    temperature: float,
    n: int,
    model: str,
    max_tokens: int,
    pbar: tqdm.tqdm,
):
    return prompt_giga_general(
        make_cot_input_prompt,
        i=i,
        cache=cache,
        gpt_query=gpt_query,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        pbar=pbar,
    )


def batch_prompt_cot_input(
    queries: list[tuple[str, str]], temperature: float, n: int, model: str, max_tokens: int, parallel: int
):
    return batch_prompt(
        prompt_cot_input,
        extract_answer_cot_input,
        queries=queries,
        temperature=temperature,
        n=n,
        model=model,
        max_tokens=max_tokens,
        parallel=parallel,
    )
