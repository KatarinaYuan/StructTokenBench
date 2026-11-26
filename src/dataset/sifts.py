import gzip
import re
from collections import defaultdict

import numpy as np


class SIFTSResidueMapping():
    """Residue-level mapping between PDB chains and UniProt, from SIFTS.

    Reads `uniprot_segments_observed.tsv.gz` from
    https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/
    Each line maps one observed segment of a PDB chain (author residue numbers
    PDB_BEG..PDB_END, with insertion codes) onto UniProt positions SP_BEG..SP_END.

    PDB author residue numbers and UniProt positions often differ (missing
    initiator methionine, signal peptides, constructs starting mid-protein,
    insertion codes), so UniProt-indexed annotations such as InterPro regions
    must be placed on PDB residues through this mapping.
    """

    def __init__(self, path):
        # (pdb_id, chain_id) -> [(uniprot_id, (num, icode), (num, icode), sp_beg, sp_end)]
        self.segments = defaultdict(list)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fin:
            header = None
            for line in fin:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if header is None:
                    header = fields
                    continue
                row = dict(zip(header, fields))
                self.segments[(row["PDB"].lower(), row["CHAIN"])].append((
                    row["SP_PRIMARY"],
                    self._parse_author_number(row["PDB_BEG"]),
                    self._parse_author_number(row["PDB_END"]),
                    int(row["SP_BEG"]),
                    int(row["SP_END"]),
                ))

    @staticmethod
    def _parse_author_number(text):
        """'52' -> (52, ''), '1H' -> (1, 'H'), '-3' -> (-3, ''); None if not a residue number"""
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", text.strip())
        if match is None:
            return None
        return int(match.group(1)), match.group(2).upper()

    def uniprot_ids(self, pdb_id, chain_id):
        """UniProt accessions SIFTS assigns to a PDB chain"""
        return {seg[0] for seg in self.segments.get((pdb_id.lower(), chain_id), [])}

    def uniprot_positions(self, pdb_chain, pdb_id, chain_id, uniprot_id):
        """UniProt position of every residue of `pdb_chain`, -1 where SIFTS gives none.

        Returns None if SIFTS does not map this chain to `uniprot_id` at all,
        i.e. the chain is not that UniProt protein.
        """
        chain_segments = [seg for seg in self.segments.get((pdb_id.lower(), chain_id), [])
                          if seg[0] == uniprot_id]
        if len(chain_segments) == 0:
            return None

        keys = [(int(num), str(icode).strip().upper())
                for num, icode in zip(pdb_chain.residue_index, pdb_chain.insertion_code)]
        key_to_pos = {key: i for i, key in enumerate(keys)}
        positions = np.full(len(keys), -1, dtype=np.int64)
        for _, beg, end, sp_beg, sp_end in chain_segments:
            if beg is None or end is None:
                continue
            span = sp_end - sp_beg
            i_beg, i_end = key_to_pos.get(beg), key_to_pos.get(end)
            if i_beg is not None and i_end is not None and i_end - i_beg == span:
                # the chain holds every residue of the segment: walk it
                # residue by residue (handles insertion codes)
                positions[i_beg:i_end + 1] = np.arange(sp_beg, sp_end + 1)
            elif beg[1] == "" and end[1] == "" and end[0] - beg[0] == span:
                # continuous numbering over the segment, but some residues are
                # absent from the parsed chain (e.g. HETATM residues such as MSE
                # are filtered out): use the segment's constant offset
                for i, (num, icode) in enumerate(keys):
                    if icode == "" and beg[0] <= num <= end[0]:
                        positions[i] = sp_beg + num - beg[0]
            # otherwise the segment cannot be placed safely and stays unmapped
        return positions

    @staticmethod
    def region_label(uniprot_positions, region_start, region_end):
        """1 for residues whose UniProt position lies in [region_start, region_end], else 0"""
        positions = np.asarray(uniprot_positions)
        return ((positions >= region_start) & (positions <= region_end)).astype(int).tolist()
