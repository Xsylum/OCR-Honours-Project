import os
import zipfile
import pandas as pd
from download_col import ensure_col_zip

def get_names_list():
    columns = [
        "col:rank",
        "col:status",
        "col:order",
        "col:class",
        "col:phylum",
        "col:genericName",
        "col:specificEpithet"
    ]

    # 1. Get the absolute path of the directory this script lives in
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = ensure_col_zip # Will download catalogue of life if not already hosted locally

    with zipfile.ZipFile(zip_path) as zf:
        # Find NameUsage.tsv even if it's nested in a subfolder inside the zip
        member = next(n for n in zf.namelist() if n.endswith("NameUsage.tsv"))
        with zf.open(member) as f:
            df = pd.read_csv(
                f, sep="\t", usecols=columns,
                on_bad_lines="skip", low_memory=False,
            )
    
    # 2. Seamlessly anchor the TSV folder path relative to this script
    tsv_path = os.path.join(current_dir, "c0d033b6-1612-448b-ac4c-b8771774d0d3", "NameUsage.tsv")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=columns,
        on_bad_lines="skip",
        low_memory=False
    )

    # Keep accepted species only
    species = df[
        (df["col:rank"] == "species") &
        (df["col:status"] == "accepted") &
        (df["col:order"] == "Coleoptera") &
        (df["col:class"] == "Insecta") &
        (df["col:phylum"] == "Arthropoda")
    ].copy()

    # Remove rows missing either part of the binomial
    species = species.dropna(
        subset=["col:genericName", "col:specificEpithet"]
    )

    # Construct "Genus species"
    species["name"] = (
        species["col:genericName"].str.strip()
        + " "
        + species["col:specificEpithet"].str.strip()
    )

    # Remove duplicates
    names = species["name"].drop_duplicates()

    # Keep simple ASCII binomial names
    names = names[
        names.str.match(r"^[A-Z][A-Za-z]+ [a-z]+$")
    ]

    # Save one name per line
    names.to_csv(
        "beetle_names.txt",
        index=False,
        header=False
    )

    print(f"Extracted {len(names):,} beetle species.")

    print("\nFirst 20:")
    for name in names.head(20):
        print(name)

    # TODO this should change to whole list, just doing head 10 for pipeline example
    output_list = []
    for name in names.head(10):
        output_list.append(name)
    return output_list