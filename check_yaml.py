from pathlib import Path
import yaml
from tqdm import tqdm

YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=YAML_LOADER)

def _structure_signature(obj):
    """
    Returns a nested "shape" of the YAML:
      - dict -> {key: signature(value), ...} with keys sorted
      - list -> ["list", signature(item0), signature(item1), ...] (uses up to a few items)
      - scalar -> type name
    """
    if isinstance(obj, dict):
        return {"__dict__": {k: _structure_signature(obj[k]) for k in sorted(obj.keys(), key=str)}}
    if isinstance(obj, list):
        # sample a few elements to avoid huge lists
        sample = obj[:3]
        return {"__list__": [_structure_signature(x) for x in sample]}
    return type(obj).__name__

def compare_yaml_structures(auth_yaml_path: str | Path, folder: str | Path, max_mismatches_to_show=20):
    auth_yaml_path = Path(auth_yaml_path)
    folder = Path(folder)

    auth_doc = _load_yaml(auth_yaml_path)
    auth_sig = _structure_signature(auth_doc)

    yaml_files = sorted(folder.rglob("*.yaml"))
    mismatches = []
    errors = []

    for p in tqdm(yaml_files, desc="Comparing structures", unit="file"):
        try:
            doc = _load_yaml(p)
            sig = _structure_signature(doc)
            if sig != auth_sig:
                mismatches.append(p)
        except Exception as e:
            errors.append((p, str(e)))

    print(f"Authentic YAML: {auth_yaml_path}")
    print(f"Folder: {folder}")
    print(f"Total YAMLs checked: {len(yaml_files)}")
    print(f"Structure mismatches: {len(mismatches)}")
    print(f"Read/parse errors: {len(errors)}")

    if mismatches:
        print("\nExamples (mismatched structure):")
        for p in mismatches[:max_mismatches_to_show]:
            print(" -", p)

    if errors:
        print("\nExamples (errors):")
        for p, err in errors[:max_mismatches_to_show]:
            print(" -", p, "=>", err)

# Example usage:
# compare_yaml_structures(
#     auth_yaml_path="/path/to/authentic.yaml",
#     folder="/path/to/folder_with_yamls"
# )


if __name__ == "__main__":
    compare_yaml_structures(
        auth_yaml_path="/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/2021_08_16_22_26_54/659/000069.yaml",
        folder="/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/CF/659")