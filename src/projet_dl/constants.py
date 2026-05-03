LABEL_MAP = {
    "normal": 0,
    "crackle": 1,
    "wheeze": 2,
    "both": 3,
}

LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
