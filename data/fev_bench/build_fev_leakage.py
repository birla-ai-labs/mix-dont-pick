"""
build_fev_leakage.py - Build fev-bench leakage flags for proxy model candidates.

Sources:
    - Chronos-Bolt, Moirai-2.0: Shchur et al. (2025), fev-bench leaderboard.
      https://huggingface.co/spaces/autogluon/fev-bench/blob/main/tables/pivot_SQL_leakage_imputed.csv
    - Lag-Llama: Rasul et al. (2023), Table 3 - pretraining corpus.
      https://arxiv.org/abs/2310.08278

Lag-Llama leakage reasoning:
    Pretrain includes ETT (H1, H2, M1) and Solar (LSTNet/NREL).
    fev-bench ETT_* and solar_* tasks overlap with these sources.
    All other fev-bench tasks have no match in Lag-Llama's pretrain corpus.

Output: data/fev_bench/fev_bench_leakage.csv
"""

import csv
from pathlib import Path

OUTPUT_PATH = Path("data/fev_bench/fev_bench_leakage.csv")

# From fev-bench pivot_SQL_leakage_imputed.csv (Chronos-Bolt and Moirai-2.0 columns only)
# Format: (task_name, chronos_bolt_leaks, moirai_2_0_leaks)
FEV_BENCH_LEAKAGE = [
    ("ETT_15T", False, True),
    ("ETT_1D", False, True),
    ("ETT_1H", False, True),
    ("ETT_1W", False, True),
    ("LOOP_SEATTLE_1D", False, True),
    ("LOOP_SEATTLE_1H", False, True),
    ("LOOP_SEATTLE_5T", False, True),
    ("M_DENSE_1D", False, True),
    ("M_DENSE_1H", False, True),
    ("SZ_TAXI_15T", False, True),
    ("SZ_TAXI_1H", False, True),
    ("australian_tourism", False, False),
    ("bizitobs_l2c_1H", False, True),
    ("bizitobs_l2c_5T", False, True),
    ("boomlet_1062", False, False),
    ("boomlet_1209", False, False),
    ("boomlet_1225", False, False),
    ("boomlet_1230", False, False),
    ("boomlet_1282", False, False),
    ("boomlet_1487", False, False),
    ("boomlet_1631", False, False),
    ("boomlet_1676", False, False),
    ("boomlet_1855", False, False),
    ("boomlet_1975", False, False),
    ("boomlet_2187", False, False),
    ("boomlet_285", False, False),
    ("boomlet_619", False, False),
    ("boomlet_772", False, False),
    ("boomlet_963", False, False),
    ("ecdc_ili", False, False),
    ("entsoe_15T", False, False),
    ("entsoe_1H", False, False),
    ("entsoe_30T", False, False),
    ("epf_be", False, False),
    ("epf_de", False, False),
    ("epf_fr", False, False),
    ("epf_np", False, False),
    ("epf_pjm", False, False),
    ("ercot_1D", False, False),
    ("ercot_1H", False, False),
    ("ercot_1M", False, False),
    ("ercot_1W", False, False),
    ("favorita_stores_1D", False, False),
    ("favorita_stores_1M", False, False),
    ("favorita_stores_1W", False, False),
    ("favorita_transactions_1D", False, True),
    ("favorita_transactions_1M", False, False),
    ("favorita_transactions_1W", False, False),
    ("fred_md_2025/cee", False, True),
    ("fred_md_2025/macro", False, True),
    ("fred_qd_2025/cee", False, False),
    ("fred_qd_2025/macro", False, False),
    ("gvar", False, False),
    ("hermes", False, False),
    ("hierarchical_sales_1D", False, True),
    ("hierarchical_sales_1W", False, True),
    ("hospital", False, True),
    ("hospital_admissions_1D", False, False),
    ("hospital_admissions_1W", False, False),
    ("jena_weather_10T", False, True),
    ("jena_weather_1D", False, True),
    ("jena_weather_1H", False, True),
    ("kdd_cup_2022_10T", False, True),
    ("kdd_cup_2022_1D", False, False),
    ("kdd_cup_2022_30T", False, False),
    ("m5_1D", False, True),
    ("m5_1M", False, False),
    ("m5_1W", False, False),
    ("proenfo_gfc12", False, True),
    ("proenfo_gfc14", False, True),
    ("proenfo_gfc17", False, True),
    ("redset_15T", False, False),
    ("redset_1H", False, False),
    ("redset_5T", False, False),
    ("restaurant", False, True),
    ("rohlik_orders_1D", False, False),
    ("rohlik_orders_1W", False, False),
    ("rohlik_sales_1D", False, False),
    ("rohlik_sales_1W", False, False),
    ("rossmann_1D", False, False),
    ("rossmann_1W", False, False),
    ("solar_1D", False, False),
    ("solar_1W", False, False),
    ("solar_with_weather_15T", False, False),
    ("solar_with_weather_1H", False, False),
    ("uci_air_quality_1D", False, False),
    ("uci_air_quality_1H", False, False),
    ("uk_covid_nation_1D/cumulative", False, False),
    ("uk_covid_nation_1D/new", False, False),
    ("uk_covid_nation_1W/cumulative", False, False),
    ("uk_covid_nation_1W/new", False, False),
    ("uk_covid_utla_1D/new", False, False),
    ("uk_covid_utla_1W/cumulative", False, False),
    ("us_consumption_1M", False, False),
    ("us_consumption_1Q", False, False),
    ("us_consumption_1Y", False, False),
    ("walmart", False, False),
    ("world_co2_emissions", False, False),
    ("world_life_expectancy", False, False),
    ("world_tourism", False, False),
]

# Lag-Llama leakage: ETT and Solar in pretrain (Rasul et al. 2023, Table 3)
LAG_LLAMA_LEAKING_PREFIXES = ["ETT_", "solar_"]


def build_leakage_csv():
    rows = []
    for task_name, chronos_bolt, moirai_2_0 in FEV_BENCH_LEAKAGE:
        lag_llama = any(task_name.startswith(p) for p in LAG_LLAMA_LEAKING_PREFIXES)
        any_leak = chronos_bolt or moirai_2_0 or lag_llama
        rows.append({
            "task_name": task_name,
            "chronos_bolt": chronos_bolt,
            "moirai_2_0": moirai_2_0,
            "lag_llama": lag_llama,
            "any_leak": any_leak,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    safe = sum(1 for r in rows if not r["any_leak"])
    print(f"Wrote {total} tasks -> {OUTPUT_PATH}")
    print(f"Safe tasks (no leakage for any model): {safe}/{total}")


if __name__ == "__main__":
    build_leakage_csv()