import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "make_protocol.py"
spec = importlib.util.spec_from_file_location("make_protocol", MODULE_PATH)
make_protocol = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(make_protocol)


def _make_audio_files(root: Path, count: int) -> None:
    for index in range(count):
        (root / f"file_{index:02d}.wav").write_bytes(b"")


def _read_protocol(output_path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        file_path, subset, label = line.split()
        rows.append((file_path, subset, label))
    return rows


def test_default_subset_remains_eval(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    _make_audio_files(root, 4)
    output = tmp_path / "protocol.txt"

    make_protocol.write_protocol(
        root_dir=root,
        output_path=output,
        num_workers=1,
    )

    rows = _read_protocol(output)
    assert len(rows) == 4
    assert {subset for _, subset, _ in rows} == {"eval"}


def test_multiple_subset_ratio_split_counts(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    _make_audio_files(root, 10)
    output = tmp_path / "protocol.txt"

    make_protocol.write_protocol(
        root_dir=root,
        output_path=output,
        subset=["train", "dev", "eval"],
        ratios=[0.5, 0.3, 0.2],
        num_workers=1,
        seed=123,
    )

    rows = _read_protocol(output)
    counts = {subset: 0 for subset in ("train", "dev", "eval")}
    for _, subset, _ in rows:
        counts[subset] += 1

    assert counts == {"train": 5, "dev": 3, "eval": 2}


def test_cli_rejects_mismatched_subset_ratio() -> None:
    try:
        make_protocol.parse_args(
            [
                "--root-dir",
                ".",
                "--subset",
                "train,dev,eval",
                "--ratio",
                "0.5,0.5",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args accepted mismatched subset/ratio lengths")
