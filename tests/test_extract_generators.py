import csv
import importlib.util
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "extract_generators.py"
spec = importlib.util.spec_from_file_location("extract_generators", MODULE_PATH)
extract_generators = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extract_generators)


def test_extract_generator_mlaad_eval() -> None:
    rel_path = "fake/de/Llasa-1B-Multilingual/file.wav"
    assert extract_generators.extract_generator(rel_path) == "Llasa-1B-Multilingual"


def test_extract_generator_mlaad_train() -> None:
    rel_path = (
        "MLAAD_v6/fake/ar/tts_models_multilingual_multi-dataset_bark/"
        "dorothy_and_wizard_oz_01_f000088.wav"
    )
    assert (
        extract_generators.extract_generator(rel_path)
        == "tts_models_multilingual_multi-dataset_bark"
    )


def test_extract_generator_mlaad_quoted() -> None:
    rel_path = '"fake/de/Chatterbox Multilingual/file.wav"'
    assert extract_generators.extract_generator(rel_path) == "Chatterbox Multilingual"


def test_extract_generator_april_synthesizers() -> None:
    rel_path = "April_Synthesizers/OmniVoice/tts/tts_default/de/00000.wav"
    assert extract_generators.extract_generator(rel_path) == "OmniVoice"


def test_extract_generator_feb_synthesizers() -> None:
    rel_path = "FEB_dataset/Synthesizers/VITS/file.wav"
    assert extract_generators.extract_generator(rel_path) == "VITS"


def test_extract_generator_feb_other() -> None:
    rel_path = "FEB_dataset/asvspoof2021_la_eval/x.wav"
    assert extract_generators.extract_generator(rel_path) == "asvspoof2021_la_eval"


def test_extract_generator_jiwon_collected() -> None:
    rel_path = "2026_April_Dataset_Jiwon_collected/WAN/1/1_1.wav"
    assert extract_generators.extract_generator(rel_path) == "WAN"


def test_extract_generator_german() -> None:
    rel_path = "tts-output/chatterbox/tts_chatterbox_full_2026/sample.wav"
    assert extract_generators.extract_generator(rel_path) == "chatterbox"


def test_extract_generator_kipot_attack_id() -> None:
    rel_path = "2025/eval/a01/1001/1.wav"
    assert extract_generators.extract_generator(rel_path) == "a01"


def test_infer_train_dataset() -> None:
    assert (
        extract_generators.infer_train_dataset(
            "MLAAD_v6/fake/ar/tts_models_multilingual_multi-dataset_bark/a.wav"
        )
        == "MLAAD_v6"
    )
    assert (
        extract_generators.infer_train_dataset("April_Synthesizers/OmniVoice/tts/de/000.wav")
        == "April_Synthesizers"
    )


def test_aggregate_generators_skips_eval_subset(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text(
        "\n".join(
            [
                "fake/de/Llasa-1B-Multilingual/a.wav eval spoof",
                "fake/bn/IndicF5/c.wav eval spoof",
                "fake/bn/IndicF5/d.wav train spoof",
                "fake/bn/IndicF5/e.wav eval bonafide",
                "only_two_fields",
            ]
        ),
        encoding="utf-8",
    )

    counts, stats = extract_generators.aggregate_generators(
        [(protocol_path, "MLAAD_v7")],
        skip_subsets=frozenset({"eval"}),
        dataset_from_path=False,
    )

    assert counts == Counter({("IndicF5", "MLAAD_v7", "train"): 1})
    assert stats.spoof_lines == 1
    assert stats.skipped_subset_lines == 2
    assert stats.non_spoof_lines == 1
    assert stats.skipped_lines == 1


def test_aggregate_generators_keeps_eval_subset_only(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text(
        "\n".join(
            [
                "MLAAD_v7/fake/de/Llasa-1B-Multilingual/a.wav eval spoof",
                "MLAAD_v7/fake/de/Llasa-1B-Multilingual/b.wav eval spoof",
                "MLAAD_v7/fake/bn/IndicF5/c.wav train spoof",
            ]
        ),
        encoding="utf-8",
    )

    counts, stats = extract_generators.aggregate_generators(
        [(protocol_path, protocol_path.stem)],
        keep_subsets=frozenset({"eval"}),
        dataset_from_path=True,
    )

    assert counts == Counter(
        {
            ("Llasa-1B-Multilingual", "MLAAD_v7", "eval"): 2,
        }
    )
    assert stats.spoof_lines == 2
    assert stats.skipped_subset_lines == 1


def test_aggregate_generators_per_eval_dataset(tmp_path: Path) -> None:
    mlaad = tmp_path / "MLAAD_v7" / "protocol.txt"
    german = tmp_path / "german_dataset_May262026" / "protocol.txt"
    mlaad.parent.mkdir()
    german.parent.mkdir()
    mlaad.write_text(
        "fake/de/Chatterbox/a.wav eval spoof\n",
        encoding="utf-8",
    )
    german.write_text(
        "tts-output/chatterbox/run/sample.wav eval spoof\n",
        encoding="utf-8",
    )

    counts, _stats = extract_generators.aggregate_generators(
        [
            (mlaad, "MLAAD_v7"),
            (german, "german_dataset_May262026"),
        ]
    )

    assert counts == Counter(
        {
            ("Chatterbox", "MLAAD_v7", "eval"): 1,
            ("chatterbox", "german_dataset_May262026", "eval"): 1,
        }
    )


def test_aggregate_generators_kipot_attack_ids(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text(
        "\n".join(
            [
                "2025/eval/a01/1001/1.wav eval spoof",
                "2025/eval/a02/1008/2.wav eval spoof",
                "2025/eval/a02/1008/3.wav eval spoof",
            ]
        ),
        encoding="utf-8",
    )

    counts, _stats = extract_generators.aggregate_generators(
        [(protocol_path, "2025_Kipot")],
    )

    assert counts == Counter(
        {
            ("a01", "2025_Kipot", "eval"): 1,
            ("a02", "2025_Kipot", "eval"): 2,
        }
    )


def test_dedupe_train_dev_counts() -> None:
    counts = Counter(
        {
            ("OmniVoice", "April_Synthesizers", "train"): 100,
            ("OmniVoice", "April_Synthesizers", "dev"): 50,
            ("VITS", "FEB_dataset", "train"): 10,
            ("MeloTTS", "FEB_dataset", "dev"): 5,
            ("MeloTTS", "MLAAD_v6", "train"): 7,
        }
    )

    rows = extract_generators.dedupe_train_dev_counts(counts)

    assert rows == [
        ("MeloTTS", 12, 7, 5, "FEB_dataset;MLAAD_v6"),
        ("OmniVoice", 150, 100, 50, "April_Synthesizers"),
        ("VITS", 10, 10, 0, "FEB_dataset"),
    ]


def test_write_deduped_train_dev_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "train_dev_no_duplicated_generators.csv"
    counts = Counter(
        {
            ("OmniVoice", "April_Synthesizers", "train"): 100,
            ("OmniVoice", "April_Synthesizers", "dev"): 50,
        }
    )

    extract_generators.write_deduped_train_dev_csv(output_path, counts)

    rows = list(csv.reader(output_path.read_text(encoding="utf-8").splitlines()))
    assert rows == [
        ["generator", "total_count", "train_count", "dev_count", "datasets"],
        ["OmniVoice", "150", "100", "50", "April_Synthesizers"],
    ]


def test_write_generator_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "generators.csv"
    counts = Counter(
        {
            ("IndicF5", "MLAAD_v7", "eval"): 3,
            ("Llasa-1B-Multilingual", "MLAAD_v7", "eval"): 2,
        }
    )

    extract_generators.write_generator_csv(output_path, counts)

    rows = list(csv.reader(output_path.read_text(encoding="utf-8").splitlines()))
    assert rows == [
        ["generator", "dataset", "subset", "count"],
        ["IndicF5", "MLAAD_v7", "eval", "3"],
        ["Llasa-1B-Multilingual", "MLAAD_v7", "eval", "2"],
    ]
