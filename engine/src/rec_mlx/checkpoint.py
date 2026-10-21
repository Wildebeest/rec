import json
from pathlib import Path


def validate(directory: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["weight_format"] != "mlx-affine-int4" or manifest["kv_format"] != "bf16":
        raise ValueError("This engine requires affine INT4 weights and BF16 KV")
    files = manifest["checkpoint_files"]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    shards = {name for name in files if name.startswith("model-") and name.endswith(".safetensors")}
    if not required.issubset(files) or not shards:
        raise ValueError("Manifest must include configuration, tokenizer and weight shards")
    # The upstream loader discovers shards by glob; an unpinned extra shard must not be loaded.
    if {p.name for p in directory.glob("model-*.safetensors")} != shards:
        raise ValueError("Checkpoint shards do not match the manifest")
    # rec verifies every file's SHA-256 before it starts the engine.
    for name in files:
        if Path(name).name != name:
            raise ValueError(f"Checkpoint filename must be a basename: {name}")
        if not (directory / name).is_file():
            raise ValueError(f"Checkpoint is missing a manifest file: {name}")
    config = json.loads((directory / "config.json").read_text())
    if config["sliding_window"] != 513:
        raise ValueError("Flint requires sliding_window=513, including the current token")
    export = config["mlx_export"]
    if (export["bits"], export["routed_expert_bits"], export["embedding_bits"]) != (4, 4, 8):
        raise ValueError("Expected INT4 projections/experts and INT8 embeddings")
    if config["quantization"] != {"group_size": 64, "bits": 4}:
        raise ValueError("Expected group-64 INT4 quantization")
    return manifest
