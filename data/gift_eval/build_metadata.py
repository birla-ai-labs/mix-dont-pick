"""
build_metadata.py - Build GIFT-Eval metadata and dataset mapping.

Source: Aksu et al. (2024), "GIFT-Eval: A Benchmark For General Time Series
Forecasting Model Evaluation", Tables 13 and 14.
https://arxiv.org/abs/2410.10393

Outputs:
    - pretrain_metadata.csv
    - eval_metadata.csv
    - dataset_mapping_v1.csv

Assumptions:
    - Univariate count = num_series x num_targets (covariates excluded)
    - Nature and Climate kept as separate domains
    - Frequency normalization: W-WED/W-SUN/W-THU/W-FRI/W-TUE -> W,
      A-DEC -> A, Q-DEC -> Q, Y -> A
    - Coverage method: PRDC if pretrain_univariate_count >= 500, else centroid
"""

import csv
from pathlib import Path


# Configuration

OUTPUT_DIR = Path("data/gift_eval")
PRDC_THRESHOLD = 500

FREQ_NORMALIZATION = {
    "W-WED": "W",
    "W-SUN": "W",
    "W-THU": "W",
    "W-FRI": "W",
    "W-TUE": "W",
    "Q-DEC": "Q",
    "A-DEC": "A",
    "Y": "A",
}

def normalize_freq(freq: str) -> str:
    return FREQ_NORMALIZATION.get(freq, freq)

# Table 14: Pretraining datasets (Aksu et al., 2024, Table 14)
# (dataset, source, domain, freq, num_series, num_targets, num_covariates, num_obs)

PRETRAIN_RAW = [
    # BuildingsBench - Energy, H
    ("BDG-2 Panther", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 105, 1, 0, 919_800),
    ("BDG-2 Fox", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 135, 1, 0, 2_324_568),
    ("BDG-2 Rat", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 280, 1, 0, 4_728_288),
    ("BDG-2 Bear", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 91, 1, 0, 1_482_312),
    ("Low Carbon London", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 713, 1, 0, 9_543_348),
    ("SMART", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 5, 1, 0, 95_709),
    ("IDEAL", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 219, 1, 0, 1_265_672),
    ("Sceaux", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 1, 1, 0, 34_223),
    ("Borealis", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 15, 1, 0, 83_269),
    ("Buildings900K", "BuildingsBench (Emami et al., 2023)", "Energy", "H", 1_792_328, 1, 0, 15_702_590_000),
    # ClimateLearn - Climate
    ("CMIP6", "ClimateLearn (Nguyen et al., 2023)", "Climate", "6H", 1_351_680, 53, 0, 1_973_453_000),
    ("ERA5", "ClimateLearn (Nguyen et al., 2023)", "Climate", "H", 245_760, 45, 0, 2_146_959_000),
    # CloudOpsTSF - CloudOps
    ("Azure VM Traces 2017", "CloudOpsTSF (Woo et al., 2023)", "CloudOps", "5T", 159_472, 1, 2, 885_522_908),
    ("Borg Cluster Data 2011", "CloudOpsTSF (Woo et al., 2023)", "CloudOps", "5T", 143_386, 2, 5, 537_552_854),
    ("Alibaba Cluster Trace 2018", "CloudOpsTSF (Woo et al., 2023)", "CloudOps", "5T", 58_409, 2, 6, 95_192_530),
    # GluonTS - mixed
    ("Taxi", "GluonTS (Alexandrov et al., 2020a)", "Transport", "30T", 67_984, 1, 0, 54_999_060),
    ("Uber TLC Daily", "GluonTS (Alexandrov et al., 2020a)", "Transport", "D", 262, 1, 0, 47_087),
    ("Uber TLC Hourly", "GluonTS (Alexandrov et al., 2020a)", "Transport", "H", 262, 1, 0, 1_129_444),
    ("Wiki-Rolling", "GluonTS (Alexandrov et al., 2020a)", "Web", "D", 47_675, 1, 0, 40_619_100),
    ("M5", "GluonTS (Alexandrov et al., 2020a)", "Sales", "D", 30_490, 1, 0, 58_327_370),
    # LargeST
    ("LargeST", "LargeST (Liu et al., 2023a)", "Transport", "5T", 42_333, 1, 0, 4_452_510_528),
    # LibCity - Transport
    ("PEMS03", "LibCity (Wang et al., 2023a)", "Transport", "5T", 358, 1, 0, 9_382_464),
    ("PEMS04", "LibCity (Wang et al., 2023a)", "Transport", "5T", 307, 3, 0, 5_216_544),
    ("PEMS07", "LibCity (Wang et al., 2023a)", "Transport", "5T", 883, 1, 0, 24_921_792),
    ("PEMS08", "LibCity (Wang et al., 2023a)", "Transport", "5T", 170, 3, 0, 3_035_520),
    ("PEMS Bay", "LibCity (Wang et al., 2023a)", "Transport", "5T", 325, 1, 0, 16_937_700),
    ("Los-Loop", "LibCity (Wang et al., 2023a)", "Transport", "5T", 207, 1, 0, 7_094_304),
    ("Beijing Subway", "LibCity (Wang et al., 2023a)", "Transport", "30T", 276, 2, 11, 248_400),
    ("SHMetro", "LibCity (Wang et al., 2023a)", "Transport", "15T", 288, 2, 0, 1_934_208),
    ("HZMetro", "LibCity (Wang et al., 2023a)", "Transport", "15T", 80, 2, 0, 146_000),
    ("Q-Traffic", "LibCity (Wang et al., 2023a)", "Transport", "15T", 45_148, 1, 0, 264_386_688),
    # SubseasonalClimateUSA - Climate
    ("Subseasonal", "SubseasonalClimateUSA (Mouatadid et al., 2023)", "Climate", "D", 862, 4, 0, 14_097_148),
    ("Subseasonal Precipitation", "SubseasonalClimateUSA (Mouatadid et al., 2023)", "Climate", "D", 862, 1, 0, 9_760_426),
    # ProEnFo - Energy
    ("Covid19 Energy", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 6, 31_912),
    ("GEF12", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 20, 1, 1, 788_280),
    ("GEF14", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 1, 17_520),
    ("GEF17", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 8, 1, 1, 140_352),
    ("PDB", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 1, 17_520),
    ("Spanish", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 1, 35_064),
    ("BDG-2 Hog", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 24, 1, 5, 421_056),
    ("BDG-2 Bull", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 41, 1, 3, 719_304),
    ("BDG-2 Cockatoo", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 5, 17_544),
    ("ELF", "ProEnFo (Wang et al., 2023b)", "Energy", "H", 1, 1, 0, 21_792),
    # Monash - mixed domains
    ("London Smart Meters", "Monash (Godahewa et al., 2021)", "Energy", "30T", 5_520, 1, 0, 166_238_880),
    ("Wind Farms", "Monash (Godahewa et al., 2021)", "Energy", "T", 337, 1, 0, 172_165_370),
    ("Wind Power", "Monash (Godahewa et al., 2021)", "Energy", "4S", 1, 1, 0, 7_397_147),
    ("Solar Power", "Monash (Godahewa et al., 2021)", "Energy", "4S", 1, 1, 0, 7_397_222),
    ("Oikolab Weather", "Monash (Godahewa et al., 2021)", "Climate", "H", 8, 1, 0, 800_456),
    ("Elecdemand", "Monash (Godahewa et al., 2021)", "Energy", "30T", 1, 1, 0, 17_520),
    ("Covid Mobility", "Monash (Godahewa et al., 2021)", "Transport", "D", 362, 1, 0, 148_602),
    ("Kaggle Web Traffic Weekly", "Monash (Godahewa et al., 2021)", "Web", "W", 145_063, 1, 0, 16_537_182),
    ("Extended Web Traffic", "Monash (Godahewa et al., 2021)", "Web", "D", 145_063, 1, 0, 370_926_091),
    ("M1 Yearly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Y", 106, 1, 0, 3_136),
    ("M1 Quarterly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Q", 198, 1, 0, 9_854),
    ("M1 Monthly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 617, 1, 0, 44_892),
    ("M3 Yearly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Y", 645, 1, 0, 18_319),
    ("M3 Quarterly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Q", 756, 1, 0, 37_004),
    ("M3 Monthly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 1_428, 1, 0, 141_858),
    ("M3 Other", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Q", 174, 1, 0, 11_933),
    ("NN5 Daily", "Monash (Godahewa et al., 2021)", "Econ/Fin", "D", 111, 1, 0, 81_585),
    ("NN5 Weekly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "W", 111, 1, 0, 11_655),
    ("Tourism Yearly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Y", 419, 1, 0, 11_198),
    ("Tourism Quarterly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Q", 427, 1, 0, 39_128),
    ("Tourism Monthly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 366, 1, 0, 100_496),
    ("CIF 2016", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 72, 1, 0, 6_334),
    ("Traffic Weekly", "Monash (Godahewa et al., 2021)", "Transport", "W", 862, 1, 0, 82_752),
    ("Traffic Hourly", "Monash (Godahewa et al., 2021)", "Transport", "H", 862, 1, 0, 14_978_112),
    ("Australian Electricity Demand", "Monash (Godahewa et al., 2021)", "Energy", "30T", 5, 1, 0, 1_153_584),
    ("Rideshare", "Monash (Godahewa et al., 2021)", "Transport", "H", 2_304, 1, 0, 859_392),
    ("Sunspot", "Monash (Godahewa et al., 2021)", "Nature", "D", 1, 1, 0, 73_894),
    ("Vehicle Trips", "Monash (Godahewa et al., 2021)", "Transport", "D", 329, 1, 0, 32_512),
    ("Weather", "Monash (Godahewa et al., 2021)", "Climate", "D", 3_010, 1, 0, 42_941_700),
    ("FRED MD", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 107, 1, 0, 76_612),
    ("Pedestrian Counts", "Monash (Godahewa et al., 2021)", "Transport", "H", 66, 1, 0, 3_130_762),
    ("Bitcoin", "Monash (Godahewa et al., 2021)", "Econ/Fin", "D", 18, 1, 0, 74_824),
    # LOTSA_Others - mixed
    ("KDD Cup 2022", "LOTSA_Others (Woo et al., 2024)", "Energy", "10T", 134, 1, 9, 4_727_519),
    ("GoDaddy", "LOTSA_Others (Woo et al., 2024)", "Econ/Fin", "M", 3_135, 2, 0, 128_535),
    ("Favorita Sales", "LOTSA_Others (Woo et al., 2024)", "Sales", "D", 111_840, 1, 0, 139_179_538),
    ("Favorita Transactions", "LOTSA_Others (Woo et al., 2024)", "Sales", "D", 54, 1, 0, 84_408),
    ("China Air Quality", "LOTSA_Others (Woo et al., 2024)", "Nature", "H", 437, 6, 0, 5_739_234),
    ("Beijing Air Quality", "LOTSA_Others (Woo et al., 2024)", "Nature", "H", 12, 11, 0, 420_768),
    ("Residential Load Power", "LOTSA_Others (Woo et al., 2024)", "Energy", "T", 271, 3, 0, 145_994_559),
    ("Residential PV Power", "LOTSA_Others (Woo et al., 2024)", "Energy", "T", 233, 3, 0, 125_338_950),
    ("CDC Fluview ILINet", "LOTSA_Others (Woo et al., 2024)", "Healthcare", "W", 75, 5, 0, 63_903),
    ("CDC Fluview WHO NREVSS", "LOTSA_Others (Woo et al., 2024)", "Healthcare", "W", 74, 4, 0, 41_760),
    ("Project Tycho", "LOTSA_Others (Woo et al., 2024)", "Healthcare", "W", 1_258, 1, 0, 1_377_707),
]


# Table 13: Evaluation datasets (Aksu et al., 2024, Table 13)
# (dataset, source, domain, freq, num_series, avg_len, min_len, max_len,
#  num_obs, target_variates, pred_len_s, win_s, pred_len_m, win_m, pred_len_l, win_l)
# None = not applicable for that prediction length

EVAL_RAW = [
    ("Jena Weather", "Autoformer (Wu et al., 2021)", "Nature", "10T", 1, 52704, 52704, 52704, 52704, 21, 48, 20, 480, 11, 720, 8),
    ("Jena Weather", "Autoformer (Wu et al., 2021)", "Nature", "H", 1, 8784, 8784, 8784, 8784, 21, 48, 19, 480, 2, 720, 2),
    ("Jena Weather", "Autoformer (Wu et al., 2021)", "Nature", "D", 1, 366, 366, 366, 366, 21, 30, 2, None, None, None, None),
    ("BizITObs - Application", "AutoMixer (Palaskar et al., 2024)", "Web/CloudOps", "10S", 1, 8834, 8834, 8834, 8834, 2, 60, 15, 600, 2, 900, 1),
    ("BizITObs - Service", "AutoMixer (Palaskar et al., 2024)", "Web/CloudOps", "10S", 21, 8835, 8835, 8835, 185535, 2, 60, 15, 600, 2, 900, 1),
    ("BizITObs - L2C", "AutoMixer (Palaskar et al., 2024)", "Web/CloudOps", "5T", 1, 31968, 31968, 31968, 31968, 7, 48, 20, 480, 7, 720, 5),
    ("BizITObs - L2C", "AutoMixer (Palaskar et al., 2024)", "Web/CloudOps", "H", 1, 2664, 2664, 2664, 2664, 7, 48, 6, 480, 1, 720, 1),
    ("Bitbrains - Fast Storage", "Grid Workloads Archive (Shen et al., 2015)", "Web/CloudOps", "5T", 1250, 8640, 8640, 8640, 10800000, 2, 48, 18, 480, 2, 720, 2),
    ("Bitbrains - Fast Storage", "Grid Workloads Archive (Shen et al., 2015)", "Web/CloudOps", "H", 1250, 721, 721, 721, 901250, 2, 48, 2, None, None, None, None),
    ("Bitbrains - rnd", "Grid Workloads Archive (Shen et al., 2015)", "Web/CloudOps", "5T", 500, 8640, 8640, 8640, 4320000, 2, 48, 18, 480, 2, 720, 2),
    ("Bitbrains - rnd", "Grid Workloads Archive (Shen et al., 2015)", "Web/CloudOps", "H", 500, 720, 720, 720, 360000, 2, 48, 2, None, None, None, None),
    ("Restaurant", "Recruit Rest. Comp. (Howard et al., 2017)", "Sales", "D", 807, 358, 67, 478, 289303, 1, 30, 1, None, None, None, None),
    ("ETT1", "Informer (Zhou et al., 2020)", "Energy", "15T", 1, 69680, 69680, 69680, 69680, 7, 48, 20, 480, 15, 720, 10),
    ("ETT1", "Informer (Zhou et al., 2020)", "Energy", "H", 1, 17420, 17420, 17420, 17420, 7, 48, 20, 480, 4, 720, 3),
    ("ETT1", "Informer (Zhou et al., 2020)", "Energy", "D", 1, 725, 725, 725, 725, 7, 30, 3, None, None, None, None),
    ("ETT1", "Informer (Zhou et al., 2020)", "Energy", "W-THU", 1, 103, 103, 103, 103, 7, 8, 2, None, None, None, None),
    ("ETT2", "Informer (Zhou et al., 2020)", "Energy", "15T", 1, 69680, 69680, 69680, 69680, 7, 48, 20, 480, 15, 720, 10),
    ("ETT2", "Informer (Zhou et al., 2020)", "Energy", "H", 1, 17420, 17420, 17420, 17420, 7, 48, 20, 480, 4, 720, 3),
    ("ETT2", "Informer (Zhou et al., 2020)", "Energy", "D", 1, 725, 725, 725, 725, 7, 30, 3, None, None, None, None),
    ("ETT2", "Informer (Zhou et al., 2020)", "Energy", "W-THU", 1, 103, 103, 103, 103, 7, 8, 2, None, None, None, None),
    ("Loop Seattle", "LibCity (Wang et al., 2023a)", "Transport", "5T", 323, 105120, 105120, 105120, 33953760, 1, 48, 20, 480, 20, 720, 15),
    ("Loop Seattle", "LibCity (Wang et al., 2023a)", "Transport", "H", 323, 8760, 8760, 8760, 2829480, 1, 48, 19, 480, 2, 720, 2),
    ("Loop Seattle", "LibCity (Wang et al., 2023a)", "Transport", "D", 323, 365, 365, 365, 117895, 1, 30, 2, None, None, None, None),
    ("SZ-Taxi", "LibCity (Wang et al., 2023a)", "Transport", "15T", 156, 2976, 2976, 2976, 464256, 1, 48, 7, 480, 1, 720, 1),
    ("SZ-Taxi", "LibCity (Wang et al., 2023a)", "Transport", "H", 156, 744, 744, 744, 116064, 1, 48, 2, None, None, None, None),
    ("M_DENSE", "LibCity (Wang et al., 2023a)", "Transport", "H", 30, 17520, 17520, 17520, 525600, 1, 48, 20, 480, 4, 720, 3),
    ("M_DENSE", "LibCity (Wang et al., 2023a)", "Transport", "D", 30, 730, 730, 730, 21900, 1, 30, 3, None, None, None, None),
    ("Solar", "LSTNet (Lai et al., 2017)", "Energy", "10T", 137, 52560, 52560, 52560, 7200720, 1, 48, 20, 480, 11, 720, 8),
    ("Solar", "LSTNet (Lai et al., 2017)", "Energy", "H", 137, 8760, 8760, 8760, 1200120, 1, 48, 19, 480, 2, 720, 2),
    ("Solar", "LSTNet (Lai et al., 2017)", "Energy", "D", 137, 365, 365, 365, 50005, 1, 30, 2, None, None, None, None),
    ("Solar", "LSTNet (Lai et al., 2017)", "Energy", "W-FRI", 137, 52, 52, 52, 7124, 1, 8, 1, None, None, None, None),
    ("Hierarchical Sales", "Mancuso et al. (2020)", "Sales", "D", 118, 1825, 1825, 1825, 215350, 1, 30, 7, None, None, None, None),
    ("Hierarchical Sales", "Mancuso et al. (2020)", "Sales", "W-WED", 118, 260, 260, 260, 30680, 1, 8, 4, None, None, None, None),
    ("M4 Yearly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "A-DEC", 22974, 37, 19, 284, 845109, 1, 6, 1, None, None, None, None),
    ("M4 Quarterly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "Q-DEC", 24000, 100, 24, 874, 2406108, 1, 8, 1, None, None, None, None),
    ("M4 Monthly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "M", 48000, 234, 60, 2812, 11246411, 1, 18, 1, None, None, None, None),
    ("M4 Weekly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "W-SUN", 359, 1035, 93, 2610, 371579, 1, 13, 1, None, None, None, None),
    ("M4 Daily", "Monash (Godahewa et al., 2021)", "Econ/Fin", "D", 4227, 2371, 107, 9933, 10023836, 1, 14, 1, None, None, None, None),
    ("M4 Hourly", "Monash (Godahewa et al., 2021)", "Econ/Fin", "H", 414, 902, 748, 1008, 373372, 1, 48, 2, None, None, None, None),
    ("Hospital", "Monash (Godahewa et al., 2021)", "Healthcare", "M", 767, 84, 84, 84, 64428, 1, 12, 1, None, None, None, None),
    ("COVID Deaths", "Monash (Godahewa et al., 2021)", "Healthcare", "D", 266, 212, 212, 212, 56392, 1, 30, 1, None, None, None, None),
    ("US Births", "Monash (Godahewa et al., 2021)", "Healthcare", "D", 1, 7305, 7305, 7305, 7305, 1, 30, 20, None, None, None, None),
    ("US Births", "Monash (Godahewa et al., 2021)", "Healthcare", "W-TUE", 1, 1043, 1043, 1043, 1043, 1, 8, 14, None, None, None, None),
    ("US Births", "Monash (Godahewa et al., 2021)", "Healthcare", "M", 1, 240, 240, 240, 240, 1, 12, 2, None, None, None, None),
    ("Saugeen", "Monash (Godahewa et al., 2021)", "Nature", "D", 1, 23741, 23741, 23741, 23741, 1, 30, 20, None, None, None, None),
    ("Saugeen", "Monash (Godahewa et al., 2021)", "Nature", "W-THU", 1, 3391, 3391, 3391, 3391, 1, 8, 20, None, None, None, None),
    ("Saugeen", "Monash (Godahewa et al., 2021)", "Nature", "M", 1, 780, 780, 780, 780, 1, 12, 7, None, None, None, None),
    ("Temperature Rain", "Monash (Godahewa et al., 2021)", "Nature", "D", 32072, 725, 725, 725, 780, 1, 30, 3, None, None, None, None),
    ("KDD Cup 2018", "Monash (Godahewa et al., 2021)", "Nature", "H", 270, 10898, 9504, 10920, 2942364, 1, 48, 20, 480, 2, 720, 2),
    ("KDD Cup 2018", "Monash (Godahewa et al., 2021)", "Nature", "D", 270, 455, 396, 455, 122791, 1, 30, 2, None, None, None, None),
    ("Car Parts", "Monash (Godahewa et al., 2021)", "Sales", "M", 2674, 51, 51, 51, 136374, 1, 12, 1, None, None, None, None),
    ("Electricity", "UCI ML Archive (Trindade, 2015)", "Energy", "15T", 370, 140256, 140256, 140256, 51894720, 1, 48, 20, 480, 20, 720, 20),
    ("Electricity", "UCI ML Archive (Trindade, 2015)", "Energy", "H", 370, 35064, 35064, 35064, 12973680, 1, 48, 20, 480, 8, 720, 5),
    ("Electricity", "UCI ML Archive (Trindade, 2015)", "Energy", "D", 370, 1461, 1461, 1461, 540570, 1, 30, 5, None, None, None, None),
    ("Electricity", "UCI ML Archive (Trindade, 2015)", "Energy", "W-FRI", 370, 208, 208, 208, 76960, 1, 8, 3, None, None, None, None),
]

# Test-side domain normalization: "Web/CloudOps" needs to match both
# "Web" and "CloudOps" in pretrain

EVAL_TO_PRETRAIN_DOMAIN_MAP = {
    "Web/CloudOps": ["Web", "CloudOps"],
    "Energy": ["Energy"],
    "Nature": ["Nature"],
    "Climate": ["Climate"],
    "Transport": ["Transport"],
    "Sales": ["Sales"],
    "Econ/Fin": ["Econ/Fin"],
    "Healthcare": ["Healthcare"],
}


def build_pretrain_metadata():
    """Build pretrain metadata from Table 14."""
    rows = []
    for (dataset, source, domain, freq, n_series, n_targets,
         n_covars, n_obs) in PRETRAIN_RAW:
        rows.append({
            "dataset": dataset,
            "source": source,
            "split": "pretrain",
            "domain": domain,
            "freq": freq,
            "freq_normalized": normalize_freq(freq),
            "num_series": n_series,
            "num_targets": n_targets,
            "num_covariates": n_covars,
            "total_univariate_series": n_series * n_targets,
            "num_obs": n_obs,
        })
    return rows


def build_eval_metadata():
    """Build eval metadata from Table 13."""
    rows = []
    for (dataset, source, domain, freq, n_series, avg_len, min_len,
         max_len, n_obs, n_variates, pl_s, w_s, pl_m, w_m,
         pl_l, w_l) in EVAL_RAW:
        rows.append({
            "dataset": dataset,
            "source": source,
            "split": "eval",
            "domain": domain,
            "freq": freq,
            "freq_normalized": normalize_freq(freq),
            "num_series": n_series,
            "num_targets": n_variates,
            "num_covariates": 0,  # not tracked in Table 13
            "total_univariate_series": n_series * n_variates,
            "num_obs": n_obs,
            "avg_series_length": avg_len,
            "min_series_length": min_len,
            "max_series_length": max_len,
            "pred_length_short": pl_s,
            "windows_short": w_s,
            "pred_length_medium": pl_m,
            "windows_medium": w_m,
            "pred_length_long": pl_l,
            "windows_long": w_l,
        })
    return rows


def build_dataset_mapping(pretrain_rows, eval_rows):
    """Build the per-dataset coverage mapping."""
    pretrain_index = {}  # (domain, freq_norm) -> list of rows
    for row in pretrain_rows:
        key = (row["domain"], row["freq_normalized"])
        pretrain_index.setdefault(key, []).append(row)

    mapping = []
    for erow in eval_rows:
        test_domain = erow["domain"]
        test_freq_norm = erow["freq_normalized"]

        pretrain_domains = EVAL_TO_PRETRAIN_DOMAIN_MAP.get(
            test_domain, [test_domain]
        )

        matched_datasets = []
        total_univariate = 0
        for pd in pretrain_domains:
            key = (pd, test_freq_norm)
            for prow in pretrain_index.get(key, []):
                matched_datasets.append(prow["dataset"])
                total_univariate += prow["total_univariate_series"]

        coverage_method = (
            "PRDC" if total_univariate >= PRDC_THRESHOLD else "centroid"
        )

        test_key = f"{erow['dataset']} / {erow['freq']}"

        notes = ""
        if total_univariate == 0:
            notes = "No pretrain data at this domain+frequency"
        elif coverage_method == "centroid":
            notes = f"Below PRDC threshold ({total_univariate} < {PRDC_THRESHOLD})"

        mapping.append({
            "test_dataset": erow["dataset"],
            "test_key": test_key,
            "test_freq": erow["freq"],
            "test_freq_normalized": test_freq_norm,
            "test_domain": test_domain,
            "test_num_variates": erow["num_targets"],
            "test_num_series": erow["num_series"],
            "pretrain_datasets": "; ".join(matched_datasets) if matched_datasets else "",
            "pretrain_num_datasets": len(matched_datasets),
            "pretrain_univariate_count": total_univariate,
            "coverage_method": coverage_method,
            "notes": notes,
        })

    return mapping


def write_csv(rows, path, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")

if __name__ == "__main__":
    pretrain_rows = build_pretrain_metadata()
    eval_rows = build_eval_metadata()
    mapping = build_dataset_mapping(pretrain_rows, eval_rows)

    write_csv(
        pretrain_rows,
        OUTPUT_DIR / "pretrain_metadata.csv",
    )
    write_csv(
        eval_rows,
        OUTPUT_DIR / "eval_metadata.csv",
    )
    write_csv(
        mapping,
        OUTPUT_DIR / "dataset_mapping_v1.csv",
    )
