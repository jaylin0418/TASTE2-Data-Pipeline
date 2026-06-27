from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(topic).strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def _infer_para_topics(run_root: Path) -> List[str]:
    para_dir = run_root / "para"
    if not para_dir.exists():
        raise FileNotFoundError(f"Cannot find para directory: {para_dir}")
    topics = sorted([p.name for p in para_dir.iterdir() if p.is_dir()])
    if not topics:
        raise ValueError(f"No topics found under: {para_dir}")
    return topics


def _infer_variant_topics(run_root: Path) -> List[str]:
    variant_dir = run_root / "variant"
    if not variant_dir.exists():
        raise FileNotFoundError(f"Cannot find variant directory: {variant_dir}")
    topics = sorted([p.name for p in variant_dir.iterdir() if p.is_dir()])
    if not topics:
        raise ValueError(f"No topics found under: {variant_dir}")
    return topics


def _detect_gpus() -> List[str]:
    # Prefer explicit CUDA_VISIBLE_DEVICES if set (treat as the pool).
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        # Preserve the user's order.
        if parts:
            return parts

    # Otherwise try nvidia-smi.
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().lower().startswith("gpu ")]
        if lines:
            return [str(i) for i in range(len(lines))]
    except Exception:
        pass

    return ["0"]


def _run_one_topic(
    python_bin: str,
    config_path: Path,
    run_root: Path,
    topic: str,
    gpu_id: str,
    logs_dir: Path,
    worker_script: Optional[str] = None,
    engine: Optional[str] = None,
) -> Tuple[str, int, str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_slugify_topic(topic)}.log"

    # Use unified run_variant_tts.py when engine is specified
    if engine:
        script = str(_script_dir() / "dialogue_v3_variant" / "run_variant_tts.py")
        cmd = [
            str(python_bin), script,
            "--engine", engine,
            "--config", str(config_path),
            "--topic", str(topic),
            "--data-root", str(run_root),
            "--device", "cuda:0",  # CUDA_VISIBLE_DEVICES 已設為單 GPU，index 永遠是 0
        ]
    else:
        script = str(_script_dir() / (worker_script or "run_topic_tts.py"))
        cmd = [
            str(python_bin), script,
            "--config", str(config_path),
            "--topic", str(topic),
            "--run-root", str(run_root),
        ]

    env = os.environ.copy()
    if not engine and not (worker_script and "para" in worker_script):
        env.setdefault("PYTHONNOUSERSITE", "1")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"GPU: {gpu_id}\n")
        log_file.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(_script_dir()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return topic, int(proc.returncode), str(log_path)


def parse_args() -> argparse.Namespace:
    default_config = _script_dir() / "conf" / "base_v2_breezy.yaml"

    p = argparse.ArgumentParser(description="Run TTS stage for multiple topics with one topic per GPU.")
    p.add_argument("--config", type=str, default=str(default_config), help="Path to pipeline config YAML")
    p.add_argument("--run-root", type=str, required=True, help="Existing session/run root")
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Optional comma-separated topic list. If omitted, inferred from run_root.",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch run_topic_tts.py",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Optional comma-separated GPU id list. If omitted, auto-detect.",
    )
    p.add_argument(
        "--topics-per-gpu",
        type=int,
        default=1,
        help=(
            "How many topics to assign to each GPU (sequentially). "
            "Example: 2 means each GPU will process 2 topics one-by-one (last group may be smaller)."
        ),
    )
    p.add_argument(
        "--concurrent-per-gpu",
        type=int,
        default=1,
        help=(
            "How many topic TTS jobs to run concurrently on the same GPU. "
            "Default 1 (safe). Example: 4 means one GPU may run 4 topics at the same time."
        ),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the overall launcher when a topic job fails",
    )
    p.add_argument(
        "--worker-script",
        type=str,
        default=None,
        help="Per-topic TTS worker script name (default: run_topic_tts.py). Ignored when --engine is set.",
    )
    p.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["indextts2", "breezyvoice", "combined"],
        help="Use run_variant_tts.py with this engine instead of a legacy worker script.",
    )
    return p.parse_args()


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _chunked(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_topics_on_gpu(
    *,
    topics: List[str],
    gpu_id: str,
    python_bin: str,
    config_path: Path,
    run_root: Path,
    logs_dir: Path,
    concurrent_per_gpu: int,
    continue_on_error: bool,
    worker_script: Optional[str] = None,
    engine: Optional[str] = None,
) -> List[Tuple[str, int, str]]:
    """Run a list of topics on a single GPU.

    When concurrent_per_gpu > 1, multiple topics are executed concurrently on the same GPU.
    If continue_on_error is False, we will not start *new* jobs after the first failure,
    but we do not force-terminate already running subprocesses.
    """
    if concurrent_per_gpu <= 1 or len(topics) <= 1:
        out: List[Tuple[str, int, str]] = []
        for topic in topics:
            topic_name, returncode, log_path = _run_one_topic(
                python_bin=python_bin,
                config_path=config_path,
                run_root=run_root,
                topic=topic,
                gpu_id=gpu_id,
                logs_dir=logs_dir,
                worker_script=worker_script,
                engine=engine,
            )
            out.append((topic_name, returncode, log_path))
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            print(f"[{status}] topic={topic_name} | gpu={gpu_id} | log={log_path}", flush=True)
            if returncode != 0 and not continue_on_error:
                break
        return out

    max_workers = min(int(concurrent_per_gpu), len(topics))
    stop_starting_new = False
    results: List[Tuple[str, int, str]] = []
    futures: List[concurrent.futures.Future] = []

    def submit_one(executor: concurrent.futures.Executor, t: str) -> None:
        futures.append(
            executor.submit(
                _run_one_topic,
                python_bin,
                config_path,
                run_root,
                t,
                gpu_id,
                logs_dir,
                worker_script,
                engine,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        it = iter(topics)
        # Prime the pool
        for _ in range(max_workers):
            try:
                submit_one(executor, next(it))
            except StopIteration:
                break

        while futures:
            done, pending = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for fut in done:
                futures.remove(fut)
                topic_name, returncode, log_path = fut.result()
                results.append((topic_name, returncode, log_path))
                status = "OK" if returncode == 0 else f"FAILED({returncode})"
                print(f"[{status}] topic={topic_name} | gpu={gpu_id} | log={log_path}", flush=True)
                if returncode != 0 and not continue_on_error:
                    stop_starting_new = True

            if stop_starting_new:
                # Do not schedule more tasks; just wait for current ones to finish.
                continue

            # Backfill with next topics
            while len(futures) < max_workers:
                try:
                    submit_one(executor, next(it))
                except StopIteration:
                    break

    return results


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    run_root = Path(args.run_root).expanduser()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    topics = _split_csv(args.topics)
    if not topics:
        worker_script_arg: Optional[str] = args.worker_script or None
        engine_arg: Optional[str] = getattr(args, "engine", None) or None
        if worker_script_arg and "para" in worker_script_arg:
            topics = _infer_para_topics(run_root)
        elif engine_arg:
            topics = _infer_variant_topics(run_root)
        else:
            raise ValueError("--topics is required unless --engine or --worker-script with 'para' is set")

    concurrent_per_gpu = int(args.concurrent_per_gpu)
    if concurrent_per_gpu <= 0:
        raise ValueError(f"concurrent-per-gpu must be > 0, got {concurrent_per_gpu}")

    gpus = _split_csv(args.gpus) or _detect_gpus()
    if not gpus:
        raise ValueError("No GPUs detected")

    logs_dir = run_root / "logs_tts"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    worker_script: Optional[str] = args.worker_script or None
    engine: Optional[str] = getattr(args, "engine", None) or None

    # Flat global queue: N_GPUs * concurrent_per_gpu workers all pull from the same pool.
    total_workers = len(gpus) * concurrent_per_gpu
    worker_gpu_ids = [gpus[i % len(gpus)] for i in range(total_workers)]

    print("=" * 80, flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Topics: {len(topics)}", flush=True)
    print(f"GPU pool: {','.join(gpus)}", flush=True)
    print(f"Total workers: {total_workers} ({len(gpus)} GPUs × {concurrent_per_gpu} per GPU)", flush=True)
    print(f"Logs: {logs_dir}", flush=True)
    print("=" * 80, flush=True)

    topic_queue: deque = deque(topics)
    queue_lock = threading.Lock()
    results: List[Tuple[str, int, str]] = []
    results_lock = threading.Lock()
    stop_flag = threading.Event()

    def single_worker(gpu_id: str) -> None:
        while not stop_flag.is_set():
            with queue_lock:
                if not topic_queue:
                    return
                topic = topic_queue.popleft()
            topic_name, rc, log_path = _run_one_topic(
                python_bin=str(args.python_bin),
                config_path=config_path,
                run_root=run_root,
                topic=topic,
                gpu_id=gpu_id,
                logs_dir=logs_dir,
                worker_script=worker_script,
                engine=engine,
            )
            status = "OK" if rc == 0 else f"FAILED({rc})"
            print(f"[{status}] topic={topic_name} | gpu={gpu_id} | log={log_path}", flush=True)
            with results_lock:
                results.append((topic_name, rc, log_path))
            if rc != 0 and not args.continue_on_error:
                stop_flag.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = [executor.submit(single_worker, gpu_id) for gpu_id in worker_gpu_ids]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    failed = [r for r in results if r[1] != 0]
    print("=" * 80, flush=True)
    print(f"Completed {len(results)} topic TTS jobs. Failed={len(failed)}", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print("=" * 80, flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
