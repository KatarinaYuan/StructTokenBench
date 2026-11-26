import os
import time
from tqdm import tqdm
from collections import Counter, defaultdict

import pandas as pd
import torch
import torch.distributed as dist

from dataset.base import BaseDataset
from dataset.cath import CATHLabelMappingDataset
from dataset.sifts import SIFTSResidueMapping
from lxml import etree as ET
import json


class InterProFunctionDataset(BaseDataset):
    """InterPro release: 2024-05-29
    """

    IPR_ENTRY_FILE = "./interpro/entry.list"
    IPR_ENTRY_PROTEIN_COUNT_FILE = "./interpro/interpro.xml"
    IDMAPPING_PDB2UNIPROT_FILE = "./interpro/pdbtosp.txt"
    PROCESSED_DETAILED_ENTRY_FILE = "./interpro/InterPro_parsed_r7_nofilter.tsv"
    # https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_segments_observed.tsv.gz
    SIFTS_SEGMENTS_FILE = "./interpro/uniprot_segments_observed.tsv.gz"

    FULL_FIELD_MAPPING = {
        "conservedsite_label": "Conserved_site",
        "binding_label": "Binding_site",
        "repeat_label": "Repeat",
        "activesite_label": "Active_site"
    }

    SPLIT_NAME = {
        "test": ["fold_test", "superfamily_test"]
    }

    SAVE_SPLIT = ["train", "validation", "fold_test", "superfamily_test"]
    
    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)
    
    def __getitem__(self, index: int):
        return BaseDataset.__getitem__(self, index)
    
    def get_target_file_name(self,):
        return os.path.join(self.data_path, f"interpro/processed_structured_{self.target_field}_{self.split}")
    
    def parse_ipr2pdb(self, report_every_n=100000):

        INTERPRO_TYPES = ["PTM", "Conserved_site", "Repeat", "Active_site", "Binding_site"]
        self.pdb_datas = {
            interpro_type: defaultdict(int) for interpro_type in INTERPRO_TYPES
        }

        # for each entry type, how many proteins 
        def handle_ipr_entry(elem):
            # Used for parsing `interpro.xml`
            ipr_entries_ = []

            ipr_id = elem.attrib["id"]
            ipr_type = elem.attrib["type"]
            if ipr_type not in self.pdb_datas.keys():
                return ipr_entries_

            pdb_pointer = elem.find("structure_db_links")
            if pdb_pointer is None:
                return ipr_entries_
            
            for child in pdb_pointer:
                # skip <taxonomy_distribution> section
                assert child.attrib["db"] == "PDB"
                pdb_id = child.attrib["dbkey"]
                ipr_entries_.append((ipr_id, ipr_type, pdb_id))
                self.pdb_datas[ipr_type][pdb_id] += 1

            elem.clear()
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]

            return ipr_entries_

        # Step 1: Parse the `interpro.xml` to retrieve the set of iprs under INTERPRO_TYPES
        file = os.path.join(self.data_path, self.IPR_ENTRY_PROTEIN_COUNT_FILE)
        ipr_entries = []
        start_time = time.time()

        context = ET.iterparse(file, tag='interpro', recover=True)
        n_records_processed = 0
        for event, elem in context:
            entries = handle_ipr_entry(elem)
            ipr_entries.extend(entries)

            n_records_processed += 1
            if n_records_processed % report_every_n == 0:
                time_used = time.time() - start_time
                print(f'Processed {n_records_processed} records; total time used: {time_used} (secs)')

        self.ipr2pdb_list = ipr_entries # (ipr_id, ipr_type, pdb_id)

    def parse_pdb_uniprot_id_mapping(self, ):

        # Step 2: parse the PDB:UniProt ID mapping
        file = os.path.join(self.data_path, self.IDMAPPING_PDB2UNIPROT_FILE)
        pdb2uniprot_list = []
        continue_flag = False
        with open(file, "r") as fin:
            for line in fin:
                l = line.strip().rstrip(",").split()
                if continue_flag:
                    # not update pdb_id
                    uniprot_nameid_list = " ".join(l)
                else:
                    assert len(l) >= 4
                    pdb_id = l[0]
                    if l[2] == "-":
                        uniprot_nameid_list = " ".join(l[3:])
                    else:
                        uniprot_nameid_list = " ".join(l[4:])
                uniprot_nameid_list = uniprot_nameid_list.split(",")
                for item in uniprot_nameid_list:
                    uniprot_id = item.strip().split(" ")[1].lstrip("()").rstrip(")")
                    pdb2uniprot_list.append((pdb_id, uniprot_id))
                if line.rstrip().endswith(","):
                    continue_flag = True
                else:
                    continue_flag = False

        cnt = {}
        for x,y in pdb2uniprot_list:
            if y not in cnt:
                cnt[y] = [x]
            else:
                cnt[y].append(x)
        self.uniprot2pdb_multi_mapping = cnt
    
    def parse_ipr_id2types(self, ):
        """Load all IPR entries
        """
        file = os.path.join(self.data_path, self.IPR_ENTRY_FILE)
        ipr_entry_names = pd.read_csv(file, sep="\t")
        self.ipr_id2type_list = ipr_entry_names[["ENTRY_AC", "ENTRY_TYPE"]].values.tolist()

    def extract_useful_features(self, ):
        self.parse_ipr_id2types()
        self.parse_ipr2pdb()
        self.parse_pdb_uniprot_id_mapping()

        file = os.path.join(self.data_path, self.PROCESSED_DETAILED_ENTRY_FILE)
        annot_df = pd.read_csv(file, sep="\t")

        self.data = []
        for i in range(len(annot_df)):
            item = annot_df.iloc[i]
            fragments = item["residue_fragments"][:-2] # remove "-S" at the end
            if item["uniprot_id"] not in self.uniprot2pdb_multi_mapping:
                continue
            for pdb_id in self.uniprot2pdb_multi_mapping[item["uniprot_id"]]:
                self.data.append({
                    "uniprot_id": item["uniprot_id"],
                    "pdb_id": pdb_id.lower(),
                    "ipr_id": item["ipr_id"],
                    "ipr_type": item["ipr_type"],
                    "fragments": fragments,
                })
    
    def associate_with_CATH_labels(self, ):
        """Associate with CATH labels
        """
        cath_data_path = os.path.join(
            self.data_path[:self.data_path.rfind("/data/")],
            "./data/CATH"
        )
        self.cath_database = CATHLabelMappingDataset(data_path=cath_data_path)
    
        for i in tqdm(range(len(self.data))):
            pdb_id = self.data[i]["pdb_id"]
            chain_id = str(None) # no chain_id provided
            ref_seq = ""
            res = self.cath_database.retrieve_labels(pdb_id, chain_id, ref_seq)
            # None: either cannot find PDB and its chain, 
            # or fail to do multi-sequence alignment
            if res is None:
                self.data[i] = None
            else:
                self.data[i]["fold_label"], self.data[i]["superfamily_label"], chain_id = res
                self.data[i]["chain_id"] = chain_id

        new_data = [x for x in self.data if x is not None]
        self.py_logger.info(f"After filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    

    def process_data_from_scratch(self, *args, **kwargs):
        assert dist.get_world_size() == 1, "dataset not preprocessed and splitted, please not to use multi-GPU training"
        
        self.extract_useful_features()
        self.associate_with_CATH_labels()

        # filter out entries without all four target labels
        self.filter_missing_labels()

        res = self.splitting_dataset()

        for i, split in enumerate(self.SAVE_SPLIT):
            target_split_file = os.path.join(self.data_path, f"interpro/{self.target_field}_{split}")
            torch.save(res[i], target_split_file)
            if split == self.split:
                self.data = res[i]

        self.py_logger.info(f"Done preprocessing, splitting and saving.")

    def filter_missing_labels(self, ):
        data_target_field = self.FULL_FIELD_MAPPING[self.target_field]
        new_data = []
        for x in self.data:
            if x["ipr_type"] != data_target_field:
                continue
            new_data.append(x)
        self.py_logger.info(f"After filtering missing labels for {self.target_field}, {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def _get_init_cnt_stats(self,):
        cnt_stats = {
            "cnt_return_none": 0,
            "cnt_wrong_chain": 0,
            "cnt_no_aligned_regions": 0,
            "cnt_aligned_regions_gapped": 0,
            "cnt_no_entity_key": 0
        }
        return cnt_stats

    def prepare_structure_loading(self):
        # InterPro regions are UniProt positions; SIFTS places them on PDB residues
        file = os.path.join(self.data_path, self.SIFTS_SEGMENTS_FILE)
        self.sifts = SIFTSResidueMapping(file)

    def load_structure(self, idx, cnt_stats):
        """Given pdb_id, chain_id
        """

        pdb_id, chain_id, residue_range, pdb = None, None, [""], None

        pdb_id = self.data[idx]["pdb_id"]
        chain_id = self.data[idx]["chain_id"]
        pdb_chain = self.get_pdb_chain(pdb_id, chain_id)
        if pdb_chain == None:
            cnt_stats["cnt_return_none"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        # fragments indexed by uniprot
        fragments = self.data[idx]["fragments"]
        fragments = fragments.split("-")
        L, R = int(fragments[0]), int(fragments[1])

        # L and R are UniProt positions, while pdb_chain.residue_index holds PDB
        # author residue numbers, which often differ (missing initiator Met,
        # signal peptides, construct offsets, insertion codes). Map every residue
        # to its UniProt position through SIFTS before labelling.
        uniprot_positions = self.sifts.uniprot_positions(
            pdb_chain, pdb_id, chain_id, self.data[idx]["uniprot_id"])
        if uniprot_positions is None:
            # the chain picked through CATH is not this UniProt protein
            cnt_stats["cnt_wrong_chain"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        # get local labels for each residue
        local_label = self.sifts.region_label(uniprot_positions, L, R)

        if sum(local_label) == 0:
            # none of the region's residues is resolved in this structure
            cnt_stats["cnt_no_aligned_regions"] += 1
            return self.NONE_RETURN_LOAD_STRUCTURE

        ret = {
            "pdb_id": pdb_id, 
            "chain_id": chain_id,
            "residue_range": residue_range,
            "pdb_chain": pdb_chain, 
            self.target_field: local_label
        }
        return ret
    