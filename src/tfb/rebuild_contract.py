"""Small immutable contracts for reproducible raw fits without teacher files."""
import hashlib
import json
import re
from pathlib import Path

from .submit import sha256


def learned_model_sha256(text):
    """Keep the complete model text; disregard only resource thread metadata."""
    text = text.replace("\r\n", "\n").rstrip("\n") + "\n"
    text, count = re.subn(r"^\[num_threads: \d+\]\n", "", text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected exactly one LightGBM num_threads parameter")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_contract(root, config, kind):
    path = Path(root) / config[kind + "_contract"]
    if sha256(path) != config[kind + "_contract_sha256"]:
        raise ValueError("Raw rebuild contract changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["format_version"] != 1 or data["kind"] != kind:
        raise ValueError("Wrong raw rebuild contract")
    return data


def verify_model(booster, expected):
    actual = learned_model_sha256(booster.model_to_string())
    if actual != expected:
        raise ValueError("Raw fitted learned trees differ from the independently reviewed model")
    return actual
