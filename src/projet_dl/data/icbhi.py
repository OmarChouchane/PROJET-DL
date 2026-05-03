from pathlib import Path


def parse_annotation_file(txt_path: Path):
    cycles = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                start_s = float(parts[0])
                end_s = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
            except (ValueError, IndexError):
                continue

            if crackle and wheeze:
                label = "both"
            elif crackle:
                label = "crackle"
            elif wheeze:
                label = "wheeze"
            else:
                label = "normal"
            cycles.append((start_s, end_s, label))
    return cycles


def load_split_file(split_path: Path):
    train_files = set()
    test_files = set()
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename, fold = parts[0], parts[1].lower()
            if fold == "train":
                train_files.add(filename)
            elif fold == "test":
                test_files.add(filename)
    return train_files, test_files
