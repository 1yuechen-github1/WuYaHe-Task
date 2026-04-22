import importlib.util
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


WEB_DIR = Path(__file__).resolve().parent
DINGLIANG_DIR = WEB_DIR.parent
RUNS_ROOT = WEB_DIR / "runs"
TMP_ROOT = WEB_DIR / ".tmp"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
TMP_ROOT.mkdir(parents=True, exist_ok=True)

GET_JIETU_MAIN = DINGLIANG_DIR / "get_jietu" / "main_debug.py"
VIS_MAIN = DINGLIANG_DIR / "vis" / "vis_debug.py"
PCA_QIANYA_MAIN = DINGLIANG_DIR / "pca-qianya" / "main.py"
PCA_HOUYA_MAIN = DINGLIANG_DIR / "pca_houya" / "main.py"
PCA_KEKONG_MAIN = DINGLIANG_DIR / "pca-kekong" / "main.py"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
RUN_CONTEXT = {}


def _log(msg: str):
    print(f"[WEB] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _import_module_from_file(module_name: str, file_path: Path):
    _log(f"import module: {module_name} <- {file_path}")
    script_dir = str(file_path.parent)
    inserted = False
    had_utils = "utils" in sys.modules
    old_utils = sys.modules.get("utils")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    # Avoid cross-module pollution for generic imports like `from utils import *`.
    if had_utils:
        sys.modules.pop("utils", None)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        if had_utils and old_utils is not None:
            sys.modules["utils"] = old_utils
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
        raise RuntimeError(f"cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Restore previous `utils` module for other pipeline stages.
        if had_utils and old_utils is not None:
            sys.modules["utils"] = old_utils
        elif had_utils and old_utils is None:
            sys.modules.pop("utils", None)
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return module


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _sorted_slice_files(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    def _k(p: Path):
        m = re.search(r"slice_(\d+)", p.stem)
        return int(m.group(1)) if m else 10**9
    return sorted(files, key=_k)


def _collect_preview_manifest(run_id: str, output_root: Path, case_id: str):
    # 预览使用“定量后”结果图，而不是原始截图
    shot_root = output_root / "pca"
    out = {}
    region_dir_map = {
        "qianya": shot_root / "pca-qianya" / case_id,
        "houya": shot_root / "pca-houya" / case_id,
        "kekong": shot_root / "pca-kekong" / case_id,
    }
    for region in ("qianya", "houya", "kekong"):
        region_dir = region_dir_map[region]
        imgs = _sorted_slice_files(region_dir)
        out[region] = [
            {
                "name": p.name,
                "url": f"/runs/{run_id}/output/pca/{region_dir.parent.name}/{case_id}/{p.name}",
            }
            for p in imgs
        ]
    return out


def _rewrite_and_run_pca_script(script_path: Path, inp_dir: Path, out_dir: Path):
    _log(f"run pca script: {script_path.name}")
    _log(f"pca inp: {inp_dir}")
    _log(f"pca out: {out_dir}")
    code = script_path.read_text(encoding="utf-8", errors="ignore")
    inp_s = str(inp_dir).replace("\\", "\\\\")
    out_s = str(out_dir).replace("\\", "\\\\")
    script_parent_s = str(script_path.parent).replace("\\", "\\\\")

    code = re.sub(r"(?m)^\s*inp\s*=\s*r?['\"].*?['\"]\s*$", f"    inp = r\"{inp_s}\"", code)
    code = re.sub(r"(?m)^\s*outp\s*=\s*r?['\"].*?['\"]\s*$", f"    outp = r\"{out_s}\"", code)
    # 强制把原脚本目录加入 sys.path，保证 `from utils import *` 可导入
    bootstrap = (
        "import sys\n"
        f"sys.path.insert(0, r\"{script_parent_s}\")\n"
    )
    code = bootstrap + code

    # 临时脚本放在原脚本目录下，确保 `from utils import *` 能正确导入同目录 utils.py
    tmp_dir = script_path.parent / ".web_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_script = tmp_dir / f"run_{script_path.parent.name}_{int(time.time() * 1000)}.py"
    tmp_script.write_text(code, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(tmp_script)],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
    )
    _log(f"pca done: {script_path.name}, returncode={proc.returncode}")
    if proc.returncode != 0:
        _log(f"pca stderr tail: {proc.stderr[-1000:]}")
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _run_pipeline(ct_file: Path, label_file: Path, output_root: Path):
    _log("pipeline start")
    _log(f"ct file: {ct_file}")
    _log(f"label file: {label_file}")
    _log(f"output root: {output_root}")
    run_base = output_root / "base"
    ct_dir = run_base / "ct"
    label_dir = run_base / "label"
    ct_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    case_id = _strip_nii_suffix(ct_file.name)
    ct_dest = ct_dir / f"{case_id}.nii.gz"
    label_dest = label_dir / f"{case_id}.nii.gz"
    shutil.copy2(ct_file, ct_dest)
    shutil.copy2(label_file, label_dest)
    _log(f"copied ct -> {ct_dest}")
    _log(f"copied label -> {label_dest}")

    qianya_dir = run_base / "screenshot" / "qianya" / case_id
    houya_dir = run_base / "screenshot" / "houya" / case_id
    kekong_dir = run_base / "screenshot" / "kekong" / case_id
    has_screenshot = qianya_dir.exists() and houya_dir.exists() and kekong_dir.exists()
    if has_screenshot:
        _log("step 1/3 skipped: screenshots already exist")
    else:
        _log("step 1/3: get_jietu screenshots")
        get_jietu = _import_module_from_file("get_jietu_main_debug_web", GET_JIETU_MAIN)
        get_jietu.spacing = 0.3
        get_jietu.process_single_ct(str(ct_dest), str(run_base), str(run_base))
        _log("step 1/3 done")

    screenshot_root = run_base / "screenshot"
    pca_root = output_root / "pca"
    pca_root.mkdir(parents=True, exist_ok=True)
    qy_len = pca_root / "pca-qianya" / case_id / "len.txt"
    hy_len = pca_root / "pca-houya" / case_id / "len.txt"
    kk_len = pca_root / "pca-kekong" / case_id / "len.txt"
    has_quant = qy_len.exists() and hy_len.exists() and kk_len.exists()
    if has_quant:
        _log("step 2/3 skipped: quantitative outputs already exist")
        pca_logs = {
            "qianya": {"ok": True, "returncode": 0, "stdout": "skipped", "stderr": ""},
            "houya": {"ok": True, "returncode": 0, "stdout": "skipped", "stderr": ""},
            "kekong": {"ok": True, "returncode": 0, "stdout": "skipped", "stderr": ""},
        }
    else:
        _log("step 2/3: run PCA scripts")
        pca_logs = {
            "qianya": _rewrite_and_run_pca_script(PCA_QIANYA_MAIN, screenshot_root / "qianya", pca_root / "pca-qianya"),
            "houya": _rewrite_and_run_pca_script(PCA_HOUYA_MAIN, screenshot_root / "houya", pca_root / "pca-houya"),
            "kekong": _rewrite_and_run_pca_script(PCA_KEKONG_MAIN, screenshot_root / "kekong", pca_root / "pca-kekong"),
        }
        for k, v in pca_logs.items():
            if not v["ok"]:
                raise RuntimeError(f"PCA {k} failed, returncode={v['returncode']}")
    for name in ("pca-qianya", "pca-houya", "pca-kekong"):
        p = pca_root / name
        file_count = sum(1 for _ in p.rglob("*")) if p.exists() else 0
        _log(f"pca output check: {p} exists={p.exists()} files={file_count}")
    _log("step 2/3 done")

    ply_path = pca_root / "vis" / f"{case_id}.ply"
    if ply_path.exists():
        _log("step 3/3: remove old vis ply and regenerate")
        ply_path.unlink()
    else:
        _log("step 3/3: vis colored mesh")
    vis = _import_module_from_file("vis_main_debug_web", VIS_MAIN)
    vis.spacing = 0.3
    # Runtime fallback for missing color-map functions in vis_debug.py.
    if not hasattr(vis, "get_clo_list_houya"):
        _log("patch vis: inject fallback get_clo_list_houya")
        def _fallback_houya(bone_len, min_val, max_val):
            den = float(max_val - min_val)
            t = 0.0 if den == 0 else max(0.0, min(1.0, float((bone_len - min_val) / den)))
            return [1.0 - t, t, 0.0]
        vis.get_clo_list_houya = _fallback_houya
    else:
        _log("vis color fn found: get_clo_list_houya")
    if not hasattr(vis, "get_clo_list_qianya"):
        _log("patch vis: inject fallback get_clo_list_qianya")
        def _fallback_qianya(bone_len, min_val, max_val):
            den = float(max_val - min_val)
            t = 0.0 if den == 0 else max(0.0, min(1.0, float((bone_len - min_val) / den)))
            return [1.0, 1.0 - t, t]
        vis.get_clo_list_qianya = _fallback_qianya
    else:
        _log("vis color fn found: get_clo_list_qianya")

    vis.process_single_ct(str(ct_dest), str(run_base), str(run_base), str(pca_root))
    if not ply_path.exists():
        raise FileNotFoundError(f"ply not found: {ply_path}")
    _log(f"step 3/3 done, ply={ply_path}")
    _log("pipeline success")

    return {
        "case_id": case_id,
        "run_base": str(run_base),
        "pca_root": str(pca_root),
        "ply_path": str(ply_path),
        "pca_logs": pca_logs,
    }


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "main.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.post("/api/run")
def api_run():
    _log("POST /api/run")
    if "ct" not in request.files or "label" not in request.files:
        _log("missing ct/label in request")
        return jsonify({"ok": False, "error": "ct/label file is required"}), 400

    ct_upload = request.files["ct"]
    label_upload = request.files["label"]
    output_dir = request.form.get("output_dir", "").strip()

    run_id = str(int(time.time() * 1000))
    _log(f"run_id={run_id}")
    serve_root = RUNS_ROOT / run_id
    serve_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(output_dir) if output_dir else (serve_root / "outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    upload_dir = serve_root / "_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ct_tmp = upload_dir / ct_upload.filename
    label_tmp = upload_dir / label_upload.filename
    ct_upload.save(str(ct_tmp))
    label_upload.save(str(label_tmp))
    _log(f"upload saved ct={ct_tmp.name}, label={label_tmp.name}")

    try:
        result = _run_pipeline(ct_tmp, label_tmp, output_root)
    except Exception as e:
        tb = traceback.format_exc()
        _log("pipeline failed")
        print(tb, flush=True)
        return jsonify({"ok": False, "error": str(e), "traceback": tb}), 500

    serve_ply = serve_root / "result.ply"
    shutil.copy2(result["ply_path"], serve_ply)
    RUN_CONTEXT[run_id] = {
        "output_root": str(output_root),
        "case_id": result["case_id"],
    }
    preview = _collect_preview_manifest(run_id, output_root, result["case_id"])
    _log(f"serve ply copied: {serve_ply}")
    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "run_root": str(output_root),
            "case_id": result["case_id"],
            "ply_url": f"/runs/{run_id}/result.ply",
            "pca_logs": result["pca_logs"],
            "preview": preview,
        }
    )


@app.get("/runs/<run_id>/<path:subpath>")
def serve_run_file(run_id: str, subpath: str):
    full = RUNS_ROOT / run_id / subpath
    if not full.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(full.parent, full.name)


@app.get("/runs/<run_id>/output/<path:subpath>")
def serve_output_file(run_id: str, subpath: str):
    ctx = RUN_CONTEXT.get(run_id)
    if not ctx:
        return jsonify({"ok": False, "error": "run context not found"}), 404
    out_root = Path(ctx["output_root"])
    full = out_root / subpath
    if not full.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(full.parent, full.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True)
