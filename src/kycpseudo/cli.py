"""Command line: synth, run, evaluate, reveal.

    kycpseudo synth    --out data/synthetic --people 12
    kycpseudo run      --in data/synthetic/images --out runs/demo --demo-keys
    kycpseudo evaluate --truth data/synthetic/truth.json --run runs/demo --demo-keys
    kycpseudo reveal   --run runs/demo --entity P-0003 --demo-keys
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import Config, load_keys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kycpseudo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a fictional KYC corpus with ground truth")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--people", type=int, default=12)
    s.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="pseudonymise a folder of scans")
    r.add_argument("--in", dest="inp", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--lama", action="store_true", help="use LaMa inpainting (optional extra)")
    r.add_argument("--demo-keys", action="store_true", help="create throwaway keys in .keys/ (testing only)")

    e = sub.add_parser("evaluate", help="score a run against synthetic ground truth")
    e.add_argument("--truth", type=Path, required=True)
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--demo-keys", action="store_true")

    v = sub.add_parser("reveal", help="key holder only: show the original behind an entity id")
    v.add_argument("--run", type=Path, required=True)
    v.add_argument("--entity", required=True)
    v.add_argument("--demo-keys", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if a.cmd == "synth":
        from .synth.generator import generate
        t = generate(a.out, a.people, a.seed)
        print(f"{len(t['documents'])} documents for {len(t['persons'])} fictional people -> {a.out}")
        return

    cfg = load_keys(Config(), demo_keys=a.demo_keys)
    if a.cmd == "run":
        from .pipeline import run
        cfg.use_lama = a.lama
        print(json.dumps(run(a.inp, a.out, cfg, workers=a.workers, use_cache=not a.no_cache), indent=2))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate, to_markdown
        m = evaluate(a.truth, a.run, cfg)
        (a.run / "metrics.json").write_text(json.dumps(m, indent=2))
        (a.run / "metrics.md").write_text(to_markdown(m))
        print(to_markdown(m))
    elif a.cmd == "reveal":
        from .stages.s9_release import open_vault
        vault = open_vault(a.run / "vault.enc", cfg)
        ent = next((e for e in vault["entities"] if e["entity_id"] == a.entity), None)
        if not ent:
            raise SystemExit(f"no entity {a.entity}")
        docs = [d["output"] for d in vault["documents"].values()
                if any(f["entity"] == a.entity for f in d["fields"])]
        print(json.dumps({"original": {k: ent[k] for k in ("given", "surname", "dob")},
                          "surrogate": vault["identities"][a.entity], "documents": docs}, indent=2))


if __name__ == "__main__":
    main()
