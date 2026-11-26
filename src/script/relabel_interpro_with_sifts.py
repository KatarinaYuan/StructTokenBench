"""Re-label the released InterPro task files through the SIFTS residue mapping.

The released `processed_structured_{label}_{split}` files of BindInt, CatInt, Con
and Rep were labelled by comparing InterPro regions (UniProt positions) with PDB
author residue numbers. This script recomputes the residue labels of every row
the same way `dataset/interpro.py` now does, and drops rows whose chain SIFTS
does not map to the row's UniProt protein, or whose region has no resolved
residue in the structure.

Usage, from src/:

    wget -P $DIR/struct_token_bench_release_data/data/functional/local/interpro \
        https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_segments_observed.tsv.gz
    python script/relabel_interpro_with_sifts.py \
        --interpro_dir $DIR/struct_token_bench_release_data/data/functional/local/interpro

Each original file is kept as `<file>.author_numbering`, and re-running the
script starts again from that copy. Tokenized caches built from the old files
(`<file>_<tokenizer>_tokenized*`) still hold the old labels: the script lists
them, and they must be deleted before training.
"""

import argparse
import glob
import os
import sys
from collections import Counter

import torch

exc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # "src/"
sys.path.append(exc_dir)
sys.path.append(os.path.join(exc_dir, "dataset"))

from sifts import SIFTSResidueMapping  # noqa: E402

LABEL_FIELDS = ["binding_label", "activesite_label", "conservedsite_label", "repeat_label"]
SPLITS = ["train", "validation", "fold_test", "superfamily_test"]
BACKUP_SUFFIX = ".author_numbering"


def relabel(rows, label_field, sifts, counts):
    kept = []
    for row in rows:
        region_start, region_end = (int(x) for x in row["fragments"].split("-"))
        positions = sifts.uniprot_positions(row["pdb_chain"], row["pdb_id"], row["chain_id"], row["uniprot_id"])
        if positions is None:
            counts["dropped: chain is not the UniProt protein"] += 1
            continue
        label = SIFTSResidueMapping.region_label(positions, region_start, region_end)
        if sum(label) == 0:
            counts["dropped: region not resolved"] += 1
            continue
        counts["relabelled" if label != [int(x) for x in row[label_field]] else "unchanged"] += 1
        row[label_field] = label
        kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interpro_dir", required=True,
                        help="the release's data/functional/local/interpro folder")
    parser.add_argument("--sifts_file", default=None,
                        help="default: <interpro_dir>/uniprot_segments_observed.tsv.gz")
    parser.add_argument("--dry_run", action="store_true", help="report the changes without writing files")
    args = parser.parse_args()

    sifts = SIFTSResidueMapping(args.sifts_file or os.path.join(args.interpro_dir, "uniprot_segments_observed.tsv.gz"))
    stale_caches = []
    for label_field in LABEL_FIELDS:
        for split in SPLITS:
            path = os.path.join(args.interpro_dir, f"processed_structured_{label_field}_{split}")
            backup = path + BACKUP_SUFFIX
            source = backup if os.path.exists(backup) else path
            rows = torch.load(source, weights_only=False)
            counts = Counter()
            kept = relabel(rows, label_field, sifts, counts)
            print(f"{label_field} {split}: {len(rows)} rows -> {len(kept)} kept; {dict(counts)}")
            if args.dry_run:
                continue
            if not os.path.exists(backup):
                os.rename(path, backup)
            torch.save(kept, path)
            stale_caches += glob.glob(path + "_*tokenized*")

    if stale_caches:
        print("These tokenized caches hold the old labels; delete them before training:")
        for cache in stale_caches:
            print("  " + cache)


if __name__ == "__main__":
    main()
