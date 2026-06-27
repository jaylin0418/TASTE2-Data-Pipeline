from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing from open_source/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from syn_para_breezy import ParaPipeline, _prepare_para_paths


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    default_config = _script_dir().parent / "conf" / "base_para_breezy.yaml"
    p = argparse.ArgumentParser(description="Run TTS stage for one para topic.")
    p.add_argument("--config", type=str, default=str(default_config), help="Path to para config YAML")
    p.add_argument("--topic", type=str, required=True, help="Topic to run TTS for")
    p.add_argument("--run-root", type=str, required=True, help="data_root for para pipeline (e.g. /work/.../syn_para_TC)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    topic = str(args.topic).strip()
    if not topic:
        raise ValueError("topic must be non-empty")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_cfg = OmegaConf.load(str(config_path))
    if "hydra" in raw_cfg:
        del raw_cfg["hydra"]

    raw_cfg["data_root"] = str(args.run_root)
    raw_cfg.scenario["topic"] = topic
    raw_cfg["stages"] = ["tts"]

    _prepare_para_paths(raw_cfg)
    logging.info("data_root: %s", raw_cfg.get("data_root"))
    ParaPipeline(raw_cfg).run()


if __name__ == "__main__":
    main()
