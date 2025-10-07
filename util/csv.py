import os, csv

def _to_float(x):
    # 兼容 tensor / numpy / 标量
    try:
        return x.item()
    except Exception:
        return float(x)

def write_csv_row(csv_path, fieldnames, row_dict):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
