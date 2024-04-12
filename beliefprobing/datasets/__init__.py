from beliefprobing.datasets.got_cities import got_cities
from beliefprobing.datasets.got_comparisons import got_comparisons
from beliefprobing.datasets.got_sp_en_trans import got_sp_en_trans
from beliefprobing.datasets.lcb_ent_bank import lcb_ent_bank
from beliefprobing.datasets.lcb_snli import lcb_snli

DATASET_REGISTRY = {
    'got_cities': got_cities,
    'got_comparisons': got_comparisons,
    'got_sp_en_trans': got_sp_en_trans,
    'lcb_ent_bank': lcb_ent_bank,
    'lcb_snli': lcb_snli,
}
