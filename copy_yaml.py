import re
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import copy

MODIFIED_RE = re.compile(r"^(?P<veh>[^_]+)_(?P<stem>.+)\.yaml$", re.IGNORECASE)
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def iter_filtered_files(root: Path, suffixes=(".yaml", ".pcd")):
    """Yield files under root (recursive) that match suffixes."""
    suffixes = tuple(s.lower() for s in suffixes)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def build_modified_index(new_path: Path):
    """
    Find flat modified YAMLs in new_path like '659_000060.yaml'
    Returns: dict veh -> list of (stem, path)
    """
    modified = {}
    for p in new_path.iterdir():
        if not p.is_file():
            continue
        m = MODIFIED_RE.match(p.name)
        if not m:
            continue
        veh = m.group("veh")
        stem = m.group("stem")
        modified.setdefault(veh, []).append((stem, p))
    return modified


def _find_original_yaml(src_veh_dir: Path, stem: str) -> Path | None:
    """Find <stem>.yaml under original_path/veh (direct first, then recursive)."""
    direct = src_veh_dir / f"{stem}.yaml"
    if direct.is_file():
        return direct
    for p in src_veh_dir.rglob(f"{stem}.yaml"):
        if p.is_file():
            return p
    return None


class AddObjectAttackModifier:
    """
    Keep original YAML as base, but append EXTRA detected vehicles into original['vehicles']
    using the SAME schema as original vehicles entries.

    If original uses:
      - center/location/extent as [x,y,z] lists => emit lists
      - center/location/extent as {x,y,z} dicts => emit dicts
      - angle as [a,b,c] list => emit [roll,pitch,yaw]
      - angle as scalar => emit yaw
    """

    @staticmethod
    def _to_float(x, default=0.0) -> float:
        return float(x) if isinstance(x, (int, float)) else float(default)

    @staticmethod
    def _to_int(x) -> int | None:
        try:
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                return int(x)
            if isinstance(x, str) and x.strip():
                return int(float(x.strip()))
        except Exception:
            return None
        return None

    @staticmethod
    def _xyz_dict(d: dict | None) -> dict:
        d = d or {}
        return {
            "x": AddObjectAttackModifier._to_float(d.get("x", 0.0)),
            "y": AddObjectAttackModifier._to_float(d.get("y", 0.0)),
            "z": AddObjectAttackModifier._to_float(d.get("z", 0.0)),
        }

    @staticmethod
    def _xyz_list(d: dict | None) -> list[float]:
        v = AddObjectAttackModifier._xyz_dict(d)
        return [v["x"], v["y"], v["z"]]

    @staticmethod
    def _angle_scalar_from_orientation(det: dict) -> float:
        ori = det.get("orientation") or {}
        return AddObjectAttackModifier._to_float(ori.get("yaw", 0.0), default=0.0)

    @staticmethod
    def _angle_list_from_orientation(det: dict) -> list[float]:
        ori = det.get("orientation") or {}
        roll = AddObjectAttackModifier._to_float(ori.get("roll", 0.0), default=0.0)
        pitch = AddObjectAttackModifier._to_float(ori.get("pitch", 0.0), default=0.0)
        yaw = AddObjectAttackModifier._to_float(ori.get("yaw", 0.0), default=0.0)
        return [roll, pitch, yaw]

    @staticmethod
    def _infer_vehicle_schema(vehicles_map: dict) -> dict:
        """
        Infer original schema from the first vehicle entry we find.
        Returns flags:
          vec_as_list: center/location/extent are lists (True) or dicts (False)
          angle_as_list: angle is list (True) or scalar (False)
        """
        schema = {"vec_as_list": False, "angle_as_list": False}

        # Grab a sample entry (like vehicle 755)
        sample = None
        for _, v in vehicles_map.items():
            if isinstance(v, dict):
                sample = v
                break

        if not isinstance(sample, dict):
            return schema  # fallback defaults

        # vectors
        c = sample.get("center")
        l = sample.get("location")
        e = sample.get("extent")
        if isinstance(c, list) or isinstance(l, list) or isinstance(e, list):
            schema["vec_as_list"] = True

        # angle
        a = sample.get("angle")
        if isinstance(a, list):
            schema["angle_as_list"] = True

        return schema

    @staticmethod
    def merge(original_doc: dict, modified_doc: dict) -> dict:
        out = copy.deepcopy(original_doc)

        vehicles_map = out.get("vehicles")
        if not isinstance(vehicles_map, dict):
            vehicles_map = {}
            out["vehicles"] = vehicles_map

        detected = modified_doc.get("detected_vehicles") or []
        if not isinstance(detected, list) or not detected:
            return out

        # Your rule: extra objects are at the tail
        n_orig = len(vehicles_map)
        if len(detected) <= n_orig:
            return out

        schema = AddObjectAttackModifier._infer_vehicle_schema(vehicles_map)
        vec_as_list = schema["vec_as_list"]
        angle_as_list = schema["angle_as_list"]

        extras = detected[n_orig:]

        for det in extras:
            if not isinstance(det, dict):
                continue

            vid = AddObjectAttackModifier._to_int(det.get("id"))
            if vid is None:
                continue

            # Convert vectors to match original formatting
            if vec_as_list:
                center = AddObjectAttackModifier._xyz_list(det.get("center"))
                extent = AddObjectAttackModifier._xyz_list(det.get("extent"))
                location = AddObjectAttackModifier._xyz_list(det.get("location"))
            else:
                center = AddObjectAttackModifier._xyz_dict(det.get("center"))
                extent = AddObjectAttackModifier._xyz_dict(det.get("extent"))
                location = AddObjectAttackModifier._xyz_dict(det.get("location"))

            # Convert angle to match original formatting
            if angle_as_list:
                angle = AddObjectAttackModifier._angle_list_from_orientation(det)
            else:
                angle = AddObjectAttackModifier._angle_scalar_from_orientation(det)

            new_vehicle_entry = {
                "angle": angle,
                "center": center,
                "extent": extent,
                "location": location,
                "speed": AddObjectAttackModifier._to_float(det.get("speed", 0.0), default=0.0),
            }

            # IMPORTANT: int key so it prints `676:` not `"676":`
            vehicles_map[vid] = new_vehicle_entry

        return out


def copy_selected_vehicles_preserve_structure(
    original_path: Path,
    new_path: Path,
    vehicles: list[str],
    overwrite_yaml: bool = False,
):
    """
    For vehicles in `vehicles`, copy .yaml and .pcd from original -> new_path/vehicle/...
    preserving relative paths under each vehicle directory.
    """
    tasks = []

    for veh in vehicles:
        src_dir = original_path / veh
        if not src_dir.is_dir():
            tqdm.write(f"[SKIP] Missing vehicle folder in original: {src_dir}")
            continue

        for src_file in iter_filtered_files(src_dir, suffixes=(".yaml", ".pcd")):
            rel = src_file.relative_to(src_dir)
            dst_file = new_path / veh / rel

            if (not overwrite_yaml) and src_file.suffix.lower() == ".yaml" and dst_file.exists():
                continue

            tasks.append((src_file, dst_file))

    for src, dst in tqdm(tasks, desc="Copy original (.yaml/.pcd)", unit="file"):
        copy_file(src, dst)


def place_modified_yamls_into_structure(
    original_path: Path,
    new_path: Path,
    modified_index: dict,
    move_modified: bool = True,
    vehicles_to_merge: set[str] | None = None,   # e.g., {"659"}
):
    vehicles_to_merge = vehicles_to_merge or set()

    mod_tasks = []
    for veh, items in modified_index.items():
        for stem, flat_yaml_path in items:
            mod_tasks.append((veh, stem, flat_yaml_path))

    for veh, stem, flat_yaml_path in tqdm(mod_tasks, desc="Place modified YAMLs", unit="file"):
        dst_yaml = new_path / veh / f"{stem}.yaml"
        src_veh_dir = original_path / veh

        # MERGE MODE (e.g., 659)
        if veh in vehicles_to_merge:
            orig_yaml_path = _find_original_yaml(src_veh_dir, stem)
            if orig_yaml_path is None:
                tqdm.write(f"[WARN] Original YAML not found for merge: {veh}/{stem}.yaml -> placing modified as-is")
                if move_modified:
                    move_file(flat_yaml_path, dst_yaml)
                else:
                    copy_file(flat_yaml_path, dst_yaml)
            else:
                with open(orig_yaml_path, "r") as f:
                    original_doc = yaml.load(f, Loader=YAML_LOADER) or {}
                with open(flat_yaml_path, "r") as f:
                    modified_doc = yaml.load(f, Loader=YAML_LOADER) or {}

                merged = AddObjectAttackModifier.merge(original_doc, modified_doc)

                dst_yaml.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_yaml, "w") as f:
                    # sort_keys=False keeps original-ish ordering; also keeps int keys as ints
                    yaml.safe_dump(merged, f, sort_keys=False)

                if move_modified:
                    flat_yaml_path.unlink(missing_ok=True)

        # DEFAULT MODE
        else:
            if move_modified:
                move_file(flat_yaml_path, dst_yaml)
            else:
                copy_file(flat_yaml_path, dst_yaml)

        # Copy matching PCD from original
        if not src_veh_dir.is_dir():
            tqdm.write(f"[WARN] No original folder for vehicle {veh}: cannot copy PCD for {stem}")
            continue

        direct_pcd = src_veh_dir / f"{stem}.pcd"
        if direct_pcd.is_file():
            copy_file(direct_pcd, new_path / veh / f"{stem}.pcd")
            continue

        found = None
        for p in src_veh_dir.rglob(f"{stem}.pcd"):
            if p.is_file():
                found = p
                break

        if found is None:
            tqdm.write(f"[WARN] Missing PCD for {veh}/{stem} in original")
            continue

        rel_pcd = found.relative_to(src_veh_dir)
        copy_file(found, new_path / veh / rel_pcd)


def main():
    original_path = Path("/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/2021_08_16_22_26_54")
    new_path = Path("/Users/dhianeifar/Desktop/Projects/CPM/Final Figure/CF")

    vehicles = ["641", "650"]  # fully copy these

    new_path.mkdir(parents=True, exist_ok=True)

    modified_index = build_modified_index(new_path)

    copy_selected_vehicles_preserve_structure(
        original_path=original_path,
        new_path=new_path,
        vehicles=vehicles,
        overwrite_yaml=False,
    )

    # Apply merge only to 659 (your malicious vehicle folder)
    place_modified_yamls_into_structure(
        original_path=original_path,
        new_path=new_path,
        modified_index=modified_index,
        move_modified=True,
        vehicles_to_merge={"659"},
    )


if __name__ == "__main__":
    main()
