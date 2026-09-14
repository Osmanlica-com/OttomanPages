import json
from collections import defaultdict
from pathlib import Path

# kadi_sicilleri, gazeteler (dergi, salname), nüfüs kayıtları




DOCUMENT_TYPES = {
    "Kadi Registry Book": "diyarbekir,usk531,bab,rsm080,eyp049,ist078,ist137,ana002,usk056,mah003,bsk063,usk002,evm001,usk396,ist018,ist022,blt002,ist003,eyp163,glt,blt001,usk017,usk051,has003,ist010,ksm019,ist020,ist024,eyp019,ksm059,ist044,ist156,bsk002,ist097,evk673,ist154, ist147, ist334, eyp037, eyp182, usk005, eyp061, tph002, eyp074, eyp138, tph002, eyp182, tph002, eyp074, eyp182, rsm040, ist191, ada001, rsm272, rsm021, mpm001",
    "Newspaper": "gazeteler,dergi,salname, hilal_ahmer",
    "Population Registry Book": "bursanufus,136_page,145_page,DefteriEvkafValideSultan,MekkeiMukerremeEvkafDefteri,481TarihliTimarDefteri",
}

DATASET_TYPE = "train"
DATASET_PATH = Path(f"dataset/OttomanPageSegmentation/{DATASET_TYPE}/layout_{DATASET_TYPE}_dataset.json")
OUTPUT_PATH = Path(f"dataset/OttomanPageSegmentation/{DATASET_TYPE}_document_types.json")


def classify_name(name: str) -> str:
    name_lower = name.lower()
    for doc_type, keywords in DOCUMENT_TYPES.items():
        for keyword in keywords.split(","):
            if keyword.strip().lower() in name_lower:
                return doc_type
    print(f"Unknown: {name}")
    return "Unknown"


def load_images(path: Path = DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["images"]


def categorize_images(images: list[dict] = None):
    if images is None:
        images = load_images()

    categorized = defaultdict(list)
    for image in images:
        doc_type = classify_name(image["name"])
        categorized[doc_type].append(image)
    return dict(categorized)


def build_classification_results(images: list[dict]=None):
    if images is None:
        images = load_images()

    results = [
        {
            "id": image["id"],
            "image_url": image["image_url"],
            "document_type": classify_name(image["name"]),
        }
        for image in images
    ]
    results.sort(key=lambda item: (item["document_type"], item["image_url"]))
    return results


def save_classification_results(
    results: list[dict],
    path: Path = OUTPUT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    images = load_images()
    rows = [
        [image["name"], classify_name(image["name"])]
        for image in images
    ]
    results = [
        {
            "id": image["id"],
            "image_url": image["image_url"],
            "document_type": doc_type,
        }
        for image, (_, doc_type) in zip(images, rows)
    ]
    results.sort(key=lambda item: (item["document_type"], item["image_url"]))
    save_classification_results(results)
    rows.sort(key=lambda row: (row[1], row[0]))

    print_table(["File Name", "Document Type"], rows)

    print()
    summary = defaultdict(int)
    for _, doc_type in rows:
        summary[doc_type] += 1

    total = len(rows)
    summary_rows = [
        [doc_type, str(count), f"{100 * count / total:.1f}%"]
        for doc_type, count in sorted(summary.items(), key=lambda x: (-x[1], x[0]))
    ]
    summary_rows.append(["Total", str(total), "100.0%"])
    print_table(["Document Type", "Count", "Percentage"], summary_rows)
    print()
    print(f"Saved classification results to {OUTPUT_PATH}")
