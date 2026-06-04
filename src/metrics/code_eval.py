from __future__ import annotations

import multiprocessing as mp
import textwrap
import traceback


def _run_code(queue: mp.Queue, code: str) -> None:
    namespace: dict = {}
    try:
        exec(code, namespace)
        queue.put({"passed": True, "error": None})
    except BaseException:
        queue.put({"passed": False, "error": traceback.format_exc(limit=5)})


def humaneval_passes(prompt: str, completion: str, test: str, entry_point: str, timeout_s: int = 5) -> dict:
    code = prompt + "\n" + completion + "\n" + test + f"\ncheck({entry_point})\n"
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_code, args=(queue, code))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"passed": False, "error": f"timeout after {timeout_s}s"}
    if queue.empty():
        return {"passed": False, "error": "no result returned"}
    result = queue.get()
    if result.get("error"):
        result["error"] = textwrap.shorten(result["error"], width=2000, placeholder=" ...")
    return result
