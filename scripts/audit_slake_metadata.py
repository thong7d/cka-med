"""Audit downloaded SLAKE metadata only; no inference, image diagnosis, or training."""

import collections
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    base = Path(__file__).resolve().parents[1] / "artifacts/source_audit/slake_metadata"
    result = {"audit_date": "2026-11-16", "source": "https://huggingface.co/datasets/BoKelvin/SLAKE", "splits": {}}
    image_sets = {}
    for split in ("train", "validation", "test"):
        path = base / f"{split}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        english = [r for r in rows if r["q_lang"] == "en"]
        # Metadata filter gives candidates, not independently confirmed CXR views.
        cxr = [r for r in english if r["modality"] == "X-Ray" and ("Chest" in r["location"] or r["location"] == "Lung")]
        visual = [r for r in english if r["base_type"] == "vqa" and r["answer_type"] == "CLOSED"]
        image_sets[split] = {r["img_name"] for r in english}
        result["splits"][split] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows_all_languages": len(rows),
            "english_qa": len(english),
            "english_images": len(image_sets[split]),
            "english_closed_visual_qa": len(visual),
            "english_kg_qa": sum(r["base_type"] == "kvqa" for r in english),
            "cxr_candidate_qa": len(cxr),
            "cxr_candidate_images": len({r["img_name"] for r in cxr}),
            "cxr_candidate_closed_visual_qa": sum(r["base_type"] == "vqa" and r["answer_type"] == "CLOSED" for r in cxr),
            "cxr_candidate_kg_qa": sum(r["base_type"] == "kvqa" for r in cxr),
            "metadata_counts": {key: dict(collections.Counter(str(r.get(key)) for r in english)) for key in ("content_type", "modality", "answer_type", "base_type", "location")},
            "patient_id_available_in_qa": any("patient_id" in r for r in rows),
        }
    result["image_name_overlap"] = {f"{a}__{b}": len(image_sets[a] & image_sets[b]) for a in image_sets for b in image_sets if a < b}
    kg_path = base / "KG.zip"
    with zipfile.ZipFile(kg_path) as archive:
        result["kg"] = {"sha256": hashlib.sha256(kg_path.read_bytes()).hexdigest(), "files": archive.namelist()}
    result["limitations"] = [
        "Image pixels not downloaded; no byte/perceptual deduplication performed.",
        "No patient-level separation established from these QA files.",
        "kvqa is a source category, not proof of joint image-knowledge dependence.",
        "Knowledge source entries are benchmark references, not independently verified medical truth.",
        "Metadata audit examined split counts and a few reference records before any model existed; original test was not used for model selection.",
    ]
    target = base / "audit_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, info in result["splits"].items():
        print(split, {k: v for k, v in info.items() if k not in ("metadata_counts", "sha256")})
    print("image_name_overlap", result["image_name_overlap"])
    print("saved", target)


if __name__ == "__main__":
    main()
