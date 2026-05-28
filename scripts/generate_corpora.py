"""Generate all synthetic corpora for the coverage-to-utility study."""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import logging
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generators import GENERATOR_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_N = 1_000_000
DEFAULT_LENGTH = 1024
DEFAULT_SEEDS = [42, 43, 44]


def _worker(gen_name, length, worker_seed, n_series, queue, worker_id):
    """Worker process: generate chunk, put result on queue."""
    try:
        gen_cls = GENERATOR_REGISTRY[gen_name]
        gen = gen_cls(length=length, random_seed=worker_seed)
        chunk = gen.generate_series(n_series)
        queue.put((worker_id, chunk))
    except Exception as e:
        queue.put((worker_id, e))


def generate_single_corpus(gen_name, seed, n, length, output_dir, workers):
    out_dir = output_dir / gen_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    series_path = out_dir / "series.npy"

    if series_path.exists():
        logger.info("SKIP  %s seed=%d (exists)", gen_name, seed)
        return {"name": gen_name, "seed": seed, "status": "skipped"}

    t0 = time.perf_counter()

    chunk_size = n // workers
    remainder = n % workers
    queue = Queue()
    procs = []

    for i in range(workers):
        n_this = chunk_size + (1 if i < remainder else 0)
        if n_this == 0:
            continue
        worker_seed = seed * 1_000 + i
        p = Process(target=_worker, args=(gen_name, length, worker_seed, n_this, queue, i))
        p.start()
        procs.append(p)

    # Collect results
    results = {}
    for _ in procs:
        wid, data = queue.get()
        if isinstance(data, Exception):
            logger.error("Worker %d failed: %s", wid, data)
            raise data
        results[wid] = data

    for p in procs:
        p.join()

    # Reassemble in worker order
    chunks = [results[i] for i in sorted(results.keys())]
    corpus = np.vstack(chunks).astype(np.float32, copy=False)
    elapsed = time.perf_counter() - t0

    np.save(series_path, corpus)
    logger.info("DONE  %s seed=%d -> %s (%.1fs)", gen_name, seed, corpus.shape, elapsed)

    return {
        "name": gen_name, "seed": seed, "status": "done",
        "elapsed_s": round(elapsed, 1), "shape": list(corpus.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic corpora")
    parser.add_argument("--generators", nargs="+",
                        default=list(GENERATOR_REGISTRY.keys()),
                        choices=list(GENERATOR_REGISTRY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--output-dir", type=str, default="data/synthetic_corpora")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timings = {}

    for gen_name in args.generators:
        for seed in args.seeds:
            result = generate_single_corpus(
                gen_name, seed, args.n, args.length, output_dir, args.workers,
            )
            if result["status"] == "done":
                timings[f"{gen_name}_seed{seed}"] = {
                    "elapsed_s": result["elapsed_s"],
                    "shape": result["shape"],
                }

    timing_path = output_dir / "generation_timings.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)
    logger.info("Timings -> %s", timing_path)


if __name__ == "__main__":
    main()