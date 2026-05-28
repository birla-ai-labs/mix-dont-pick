"""Allows loading and evaluation with the GIFT Eval evaluation datasets.

Provides an interface to access the datasets through the Dataset class.
Please refer to the GIFT Eval documentation for more details.
"""
from __future__ import annotations

import logging
import math
import pickle as pkl
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import pandas as pd
import pyarrow.compute as pc
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import Map
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality, norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose
from tqdm import tqdm

from config import (
    GIFTEVAL_EVAL_DATA_DIR,
    GIFTEVAL_DATASETS,
    GIFTEVAL_DATASETS_PROPERTIES,
    MEDIUM_LONG_GIFTEVAL_DATASETS,
)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from gluonts.dataset import DataEntry
    from gluonts.ev.metrics import BaseMetricDefinition
    from gluonts.model.predictor import RepresentablePredictor

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
}

NAIVE_MODELS = [
    "naive",
    "seasonal_naive",
    "auto_ets",
    "auto_arima",
    "auto_theta",
]

class Term(Enum):
    """Enum representing the forecasting horizon term."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        """Returns the multiplier for the forecasting horizon based on the term."""
        if self == Term.SHORT:
            return 1
        if self == Term.MEDIUM:
            return 10
        if self == Term.LONG:
            return 15
        return None


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """Return a DataEntry with the start time as an item."""
    data_entry["start"] = data_entry["start"].item()
    return data_entry

class MultivariateToUnivariate(Transformation):
    """Transformation to convert multivariate time series data to univariate."""

    def __init__(self, field: str) -> None:
        """Initialize the transformation with the specified field."""
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False,  # noqa: ARG002, FBT001, FBT002 following the GIFT Eval Code
    ) -> Iterator:
        """Apply the transformation to each data entry in the iterable."""
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for val_id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(val_id)
                yield univariate_entry

class Dataset:
    """Dataset class for loading and processing GIFT Eval datasets."""

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
    ) -> None:
        """Initialize the Dataset with the given parameters.

        Args:
            name (str): Name of the dataset.
            storage_path (str | Path): Path to the dataset storage.
            term (Term | str): Forecasting horizon term, can be a Term enum or string.

        """
        if isinstance(term, str):
            term = Term(term.lower())
        start = time.time()
        self.hf_dataset = datasets.load_from_disk(
                            str(GIFTEVAL_EVAL_DATA_DIR / name),
                        ).with_format("numpy")
        msg = f"Loaded dataset {name} in {time.time() - start:.2f} seconds."
        logger.debug(msg)
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        to_univariate = (self.target_dim != 1)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset,
            )

        self.term = Term(term)
        self.name = name

    @cached_property
    def prediction_length(self) -> int:
        """Calculate the prediction length based on the dataset frequency and term."""
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        """Get the frequency of the dataset."""
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        """Get the dimension of the target variable."""
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        """Get the dimension of the past dynamic features."""
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        if (
            len(
                (
                    past_feat_dynamic_real := self.hf_dataset[0][
                        "past_feat_dynamic_real"
                    ]
                ).shape,
            )
            > 1
        ):
            return past_feat_dynamic_real.shape[0]
        return 1

    @cached_property
    def windows(self) -> int:
        """Return the number of windows based on the dataset and prediction length."""
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1),
                ),
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        """Calculate the total length of all series in the dataset."""
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column("target")),
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        """Return the training dataset split from the full dataset."""
        training_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        """Return the validation dataset split from the full dataset."""
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        """Return the test dataset split from the full dataset."""
        _, test_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows,
        )
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )

def _evaluate_single_dataset(  # noqa: PLR0913
    ds_config: str,                # e.g., "electricity/15T/short"
    output_dir: Path,
    metrics: list,
    metrics_name: list,
    batch_size: int,
    *,
    model_name: str,
    allow_nan_forecast: bool,
    delete_pkl: bool,
) -> dict:
    """Evaluate a single dataset and return the results as a dictionary."""
    ds_id = ds_config.replace("/", "-")
    pkl_path = Path(output_dir) / f"{ds_id}.pkl"

    with pkl_path.open("rb") as f:
        ds_dict = pkl.load(f)  # noqa: S301

    ds_name = ds_dict["ds_name"]
    ds_key = ds_dict["ds_key"]
    forecasts = ds_dict["forecasts"]
    ds_term = ds_dict["term"]

    data = Dataset(name=ds_name, term=ds_term)
    if batch_size is None:
        res = evaluate_forecasts(
            forecasts=forecasts,
            test_data=data.test_data,
            metrics=metrics,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=allow_nan_forecast,
            seasonality=get_seasonality(data.freq),
        )
    else:
        res = evaluate_forecasts(
            forecasts=forecasts,
            test_data=data.test_data,
            metrics=metrics,
            axis=None,
            batch_size=batch_size,
            mask_invalid_label=True,
            allow_nan_forecast=allow_nan_forecast,
            seasonality=get_seasonality(data.freq),
        )

    result_row = {
        "dataset": ds_config,
        "model": model_name,
        **{name: res[name][0] for name in metrics_name},
        "domain": GIFTEVAL_DATASETS_PROPERTIES[ds_key]["domain"],
        "num_variates": GIFTEVAL_DATASETS_PROPERTIES[ds_key]["num_variates"],
    }

    if delete_pkl:
        pkl_path.unlink(missing_ok=True)
    return result_row

class Evaluator:
    """Evaluator class for GIFT Eval datasets."""

    def __init__(
            self,
            model_name: str,
            output_dir: str | Path,
            predictor: RepresentablePredictor,
            predictor_args: dict | None = None,
            csv_name: str = "all_results.csv",
        ) -> None:
        """Initialize the Evaluator with the output path.

        Args:
            model_name (str): Name of the model being evaluated.
            output_dir (str | Path): Directory to save the evaluation results.
            predictor (RepresentablePredictor): The predictor class to be used for evaluation.
            predictor_args (dict | None): Arguments to be passed to the predictor.
            csv_name (str): Name of the CSV file to save the results.

        """  # noqa: E501
        self.predictor = predictor
        self.predictor_args = predictor_args
        self.csv_name = csv_name
        self.model_name = model_name
        self.output_dir = Path(output_dir) / self.model_name

        self.data_path = self.output_dir / self.csv_name

        if Path.exists(self.data_path):
            self.datasets_done = pd.read_csv(self.data_path)[
                "dataset"
            ].unique().tolist()
        else:
            self.datasets_done = []

    def iterate_datasets(self) -> None:
        """Predict results from all datasets for a given model."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
        }

        for ds_name in tqdm(GIFTEVAL_DATASETS, total=len(GIFTEVAL_DATASETS)):
            ds_key = ds_name.split("/")[0]
            terms = ["short", "medium", "long"]
            try:
                for term in terms:
                    if (
                        term in {"medium", "long"}
                    ) and ds_name not in MEDIUM_LONG_GIFTEVAL_DATASETS:
                        continue
                    if "/" in ds_name:
                        ds_key = ds_name.split("/")[0]
                        ds_freq = ds_name.split("/")[1]
                        ds_key = ds_key.lower()
                        ds_key = pretty_names.get(ds_key, ds_key)
                    else:
                        ds_key = ds_name.lower()
                        ds_key = pretty_names.get(ds_key, ds_key)
                        ds_freq = GIFTEVAL_DATASETS_PROPERTIES[ds_key]["frequency"]
                    ds_config = f"{ds_key}/{ds_freq}/{term}"

                    if ds_config in self.datasets_done:
                        logger.debug(
                            "Skipping %s as it has already been evaluated.",
                            ds_config,
                        )
                        continue

                    logger.info("Processing dataset: %s for term: %s", ds_name, term)

                    dataset = Dataset(name=ds_name, term=term)
                    season_length = get_seasonality(dataset.freq)
                    kwargs = {
                        "prediction_length": dataset.prediction_length,
                        "freq": dataset.freq,
                        "season_length": season_length,
                        "target_dim": dataset.target_dim,
                        "past_feat_dynamic_real_dim": dataset.past_feat_dynamic_real_dim,  # noqa: E501
                        "windows": dataset.windows,
                        "sum_series_length": dataset.sum_series_length,
                        "training_dataset": dataset.training_dataset,
                        "validation_dataset": dataset.validation_dataset,
                        "test_data": dataset.test_data,
                    }
                    if self.predictor_args is not None:
                        kwargs.update(self.predictor_args)

                    predictor = self.predictor(**kwargs)
                    start = time.time()
                    try:
                        forecasts = list(predictor.predict(dataset.test_data.input))
                    except Exception:
                        logger.exception(
                            "Error occurred while predicting for %s",
                            ds_config,
                        )
                        continue

                    output_path = self.output_dir / f"{ds_config.replace('/', '-')}.pkl"
                    with output_path.open("wb") as f:
                        pkl.dump(
                            {
                                "ds_name": ds_name,
                                "ds_key": ds_key,
                                "forecasts": forecasts,
                                "term": term,
                            },
                            f,
                        )

                    logger.info(
                        "Prediction completed for %d items in %.2f seconds.",
                        len(dataset.test_data),
                        time.time() - start,
                    )
            except Exception:
                logger.exception("Error processing dataset: %s", ds_name)
                continue

    def evaluate_datasets(
        self,
        metrics: list[BaseMetricDefinition] | str = "default",
        batch_size: int = 1024,
        *,
        allow_nan_forecast: bool = False,
        save_forecasts: bool = False,
    ) -> pd.DataFrame:
        """Evaluate the predictions for all datasets and save the results.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation results.

        """
        if metrics == "default":
            metrics = [
                MSE(forecast_type="mean"),
                MSE(forecast_type=0.5),
                MAE(),
                MASE(),
                MAPE(),
                SMAPE(),
                MSIS(),
                RMSE(),
                NRMSE(),
                ND(),
                MeanWeightedSumQuantileLoss(
                    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                ),
            ]

        metrics_name = [metric.__call__().name for metric in metrics]
        start = time.time()

        all_ds_names = [
            ds_config for ds_config in self.output_dir.iterdir()
            if ds_config.suffix == ".pkl" and ds_config.stem not in self.datasets_done
        ]
        all_ds_names = [p.stem for p in all_ds_names]
        all_ds_ids = [name.replace("-", "/") for name in all_ds_names]

        logger.info(
            "Starting evaluation of %d datasets in parallel...",
            len(all_ds_ids),
        )

        partial_worker = partial(
            _evaluate_single_dataset,
            output_dir=self.output_dir,
            metrics=metrics,
            metrics_name=metrics_name,
            batch_size=batch_size,
            allow_nan_forecast=allow_nan_forecast,
            model_name=self.model_name,
            delete_pkl=not save_forecasts,
        )

        all_results = []
        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = {
                executor.submit(partial_worker, ds_id): ds_id
                for ds_id in all_ds_ids
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                result_row = future.result()
                if result_row:
                    all_results.append(result_row)
                    logger.debug(
                        "Saved Results: %s, Model: %s",
                        result_row.get("dataset", "unknown"),
                        self.model_name,
                    )

        csv_path = self.output_dir / self.csv_name
        if csv_path.exists():
            results_df = pd.read_csv(csv_path)
        else:
            results_df = pd.DataFrame(
                columns=[
                    "dataset",
                    "model",
                    *metrics_name,
                    "domain",
                    "num_variates",
                ],
            )
        if all_results:
            results_df = pd.concat(
                [results_df, pd.DataFrame(all_results)],
                ignore_index=True,
            )
            results_df.to_csv(csv_path, index=False)

        logger.info(
            "Evaluation completed in %.2f seconds. Results saved to %s.",
            time.time() - start,
            csv_path,
        )

        return results_df