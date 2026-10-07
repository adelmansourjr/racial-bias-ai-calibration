"""Construct a small, reproducible matched-name audit from the supplied datasets."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
JOBS = [
    (175485704, "INFORMATION-TECHNOLOGY", "PHP"),
    (103860943, "SALES", "customer service"),
    (258868844, "SALES", "sales"),
    (56924323, "ENGINEERING", "license"),
    (263583866, "HEALTHCARE", "registered nurse"),
    (1183148438, "TEACHER", "children"),
]


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def excerpt(text, keyword, word_limit=135):
    words = clean(text).split()
    if len(words) <= word_limit:
        return " ".join(words)
    lower = clean(text).lower()
    at = lower.find(keyword.lower())
    if at < 0:
        return " ".join(words[:word_limit])
    token_index = len(lower[:at].split())
    start = max(0, token_index - 40)
    return " ".join(words[start : start + word_limit])


def main():
    names_path = ROOT / "name_bank.json"
    names = json.loads(names_path.read_text())
    with (DATA / "Resume.csv").open() as f:
        resumes = list(csv.DictReader(f))
    jobs = {row["row"]["job_id"]: row["row"] for row in
            json.loads((DATA / "jobs-first-rows.json").read_text())["rows"]}
    resumes.sort(key=lambda row: int(row["ID"]))
    used_ids, used_hashes = set(), set()
    cases = []
    for ji, (job_id, category, keyword) in enumerate(JOBS):
        job = jobs[job_id]
        assert keyword.lower() in job["description"].lower()
        selected = []
        for positive in (True, False):
            for row in resumes:
                text = clean(row["Resume_str"])
                digest = hashlib.sha256(text.encode()).hexdigest()
                if row["ID"] in used_ids or digest in used_hashes:
                    continue
                present = keyword.lower() in text.lower()
                if present != positive:
                    continue
                if positive and row["Category"] != category:
                    continue
                if not positive and row["Category"] == category:
                    continue
                selected.append((row, positive, digest))
                used_ids.add(row["ID"]); used_hashes.add(digest)
                if sum(pos == positive for _, pos, _ in selected) == 2:
                    break
        if len(selected) != 4:
            raise ValueError(f"Need two positive and negative evidence controls for {keyword}")
        for ci, (row, positive, digest) in enumerate(selected):
            resume_text = excerpt(row["Resume_str"], keyword)
            label = int(keyword.lower() in resume_text.lower())
            assert label == positive
            split = "calibration" if ci == 0 or (ji < 2 and ci == 3) else "test"
            case_id = f"job{job_id}_resume{row['ID']}"
            variants = []
            for name in names:
                variants.append({**name, "text": f"NAME: {name['name']}\n{resume_text}"})
            cases.append({
                "case_id": case_id, "split": split,
                "resume_id": row["ID"], "resume_source_sha256": digest,
                "resume_category_for_sampling_only": row["Category"],
                "job_id": job_id, "job_title": job["title"],
                "job_description": excerpt(job["description"], keyword, 90),
                "resume_text": resume_text, "requirement_keyword": keyword,
                "evidence_label": label, "qualification_label": None,
                "variants": variants,
            })
    metadata = {
        "resume_source": "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset",
        "jobs_source": "https://huggingface.co/datasets/datastax/linkedin_job_listings",
        "resume_rows_downloaded": len(resumes), "job_rows_downloaded": len(jobs),
        "source_file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [DATA / "Resume.csv", DATA / "jobs-first-rows.json"]},
        "name_reference": "https://www.nber.org/papers/w9873",
        "design": "24 distinct public resume excerpts, 6 jobs, fixed matched name variants; 8 calibration and 16 test cases",
        "evidence_label_definition": "Case-insensitive literal occurrence of the named keyword in the supplied resume excerpt; a diagnostic evidence task, not hiring ground truth",
        "qualification_label_definition": "Not available; no balanced accuracy for real hiring suitability is asserted",
        "limitations": ["Small convenience sample", "Shortened texts", "Controlled name association only", "No real race labels or applicant outcomes", "Within each name group only two names", "Related job contexts are repeated"],
    }
    output = {"metadata": metadata, "cases": cases}
    (ROOT / "cases.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "calibration": sum(c['split']=='calibration' for c in cases),
                      "test": sum(c['split']=='test' for c in cases), "variants_per_case":len(names),
                      "test_evidence_positives":sum(c['evidence_label'] for c in cases if c['split']=='test')}, indent=2))


if __name__ == "__main__":
    main()
