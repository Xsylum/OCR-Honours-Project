import os
import zipfile
import pandas as pd

try:
    from .download_col import ensure_col_zip          # when imported as CatalogueOfLife.extract_beetles
except ImportError:
    from download_col import ensure_col_zip           # when this file is run directly


def get_names_list(limit=10):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    names_path = os.path.join(current_dir, "beetle_names.txt")

    if os.path.exists(names_path):                     # cached: skip the huge TSV entirely
        with open(names_path, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        columns = ["col:rank", "col:status", "col:order", "col:class",
                   "col:phylum", "col:genericName", "col:specificEpithet"]
        zip_path = ensure_col_zip()                    # downloads Catalogue of Life if it isn't on disk
        parts = []
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if n.endswith("NameUsage.tsv"))
            with zf.open(member) as f:
                for chunk in pd.read_csv(f, sep="\t", usecols=columns, dtype=str,
                                         on_bad_lines="skip", chunksize=200_000):
                    keep = chunk[
                        (chunk["col:rank"] == "species") &
                        (chunk["col:status"] == "accepted") &
                        (chunk["col:order"] == "Coleoptera") &
                        (chunk["col:class"] == "Insecta") &
                        (chunk["col:phylum"] == "Arthropoda")
                    ].dropna(subset=["col:genericName", "col:specificEpithet"])
                    parts.append(keep["col:genericName"].str.strip() + " " +
                                 keep["col:specificEpithet"].str.strip())
        series = pd.concat(parts).drop_duplicates()
        series = series[series.str.match(r"^[A-Z][A-Za-z]+ [a-z]+$")]   # simple ASCII binomials
        series.to_csv(names_path, index=False, header=False)
        names = series.tolist()

    print(f"{len(names):,} beetle species available.")
    return names[:limit]