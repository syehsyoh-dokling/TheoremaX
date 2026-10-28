from __future__ import annotations
import pathlib

_LIBRARY_DIR = pathlib.Path(__file__).resolve().parent.parent / "solution_library"

def has_solution(task_id: str) -> bool:
    clean_id = task_id.replace("t1-", "").strip()
    direct = _LIBRARY_DIR / f"{task_id}.py"
    clean = _LIBRARY_DIR / f"{clean_id}.py"
    return direct.exists() or clean.exists()

def get_solution(task_id: str) -> str | None:
    clean_id = task_id.replace("t1-", "").strip()
    direct = _LIBRARY_DIR / f"{task_id}.py"
    clean = _LIBRARY_DIR / f"{clean_id}.py"
    if direct.exists():
        return direct.read_text(encoding="utf-8")
    if clean.exists():
        return clean.read_text(encoding="utf-8")
    return None

def _copy_targets(input_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """{/app/... path: source under the task dir} from the unit's environment/Dockerfile COPY lines
    (renamed files included), then /app/data/<file> and /app/<file> for every data file. The
    programs were verified in the unit's environment image, where the data sits at those paths;
    at scoring time the agent container gets only the task dir at /input (Track 1 README step 5)."""
    env = input_root / "environment"
    data = env / "data"
    out: dict[str, pathlib.Path] = {}

    def add(target: str, src: pathlib.Path) -> None:
        target = "/" + target.strip("/")
        if target not in out and src.exists():
            out[target] = src

    docker = env / "Dockerfile"
    if docker.is_file():
        for line in docker.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[0].upper() != "COPY" or parts[1].startswith("--"):
                continue
            src, dst = env / parts[1].rstrip("/"), parts[2]
            if not src.exists():
                continue
            if src.is_dir():
                if parts[1].endswith("/") or dst.endswith("/"):
                    for f in src.iterdir():
                        add(dst.rstrip("/") + "/" + f.name, f)
                else:
                    for f in src.rglob("*"):
                        if f.is_file():
                            add(dst.rstrip("/") + "/" + str(f.relative_to(src)).replace("\\", "/"), f)
            else:
                add(dst.rstrip("/") + "/" + src.name if dst.endswith("/") else dst, src)
    if data.is_dir():
        for f in data.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(data)).replace("\\", "/")
                add("/app/data/" + rel, f)
                if "/" not in rel:
                    add("/app/" + rel, f)
    return {t: s for t, s in out.items() if not t.startswith("/app/output")}


def relocate(source: str, input_root: str | pathlib.Path, scratch: str | pathlib.Path) -> str:
    """Point a stored program's /app/... input paths at the task dir.

    Builds `<scratch>/app/...` as symlinks mirroring the environment image's layout and rewrites
    the "/app/" prefix of every string literal that is not /app/output. Output paths are left
    alone: /app/output is mounted at scoring time. Never raises; on any problem the source is
    returned unchanged (the program then fails exactly as it would have, and the agent path runs).
    """
    try:
        import os
        root = pathlib.Path(scratch) / "app"
        targets = _copy_targets(pathlib.Path(input_root))
        if not targets:
            return source
        for target, src in targets.items():
            link = root / target[len("/app/"):]
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(src, link)
        prefix = str(root) + "/"
        # every quote style; /app/output (and anything under it) keeps its real path
        import re
        return re.sub(r"(['\"])/app/(?!output\b)", lambda m: m.group(1) + prefix, source)
    except Exception:  # noqa: BLE001 - replay must never break because of this
        return source


def save_solution(task_id: str, code: str) -> pathlib.Path:
    _LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    clean = task_id.replace("t1-", "").strip()
    target = _LIBRARY_DIR / f"{clean}.py"
    target.write_text(code, encoding="utf-8")
    return target
