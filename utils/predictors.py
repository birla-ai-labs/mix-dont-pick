"""
utils/predictors.py — Predictor wrappers for TSTR evaluation.

Both implement the interface expected by utils/gift_eval.Evaluator:
    __init__(**kwargs) where kwargs include prediction_length, freq,
    season_length, target_dim, test_data, etc.
    predict(inputs) -> Iterator[Forecast]
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

class ChronosPredictor:
    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        num_samples: int = 20,
        quantile_levels: list[float] | None = None,
        **_,
    ) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )

    def predict(self, dataset):
        import torch
        import numpy as np
        from gluonts.model.forecast import QuantileForecast

        q_levels = np.array(self.quantile_levels, dtype=float)
        q_keys   = [str(round(q, 2)) for q in q_levels]
        batch_size = 128

        contexts, start_dates, item_ids = [], [], []
        for idx, item in enumerate(dataset):
            series = np.asarray(item["target"], dtype=np.float32)
            contexts.append(torch.tensor(series, dtype=torch.float32))
            start_dates.append(item["start"] + len(series))
            item_ids.append(str(item.get("item_id", idx)))

        forecasts_out = []
        i = 0
        while i < len(contexts):
            j = min(i + batch_size, len(contexts))
            try:
                with torch.inference_mode():
                    q, _ = self.pipeline.predict_quantiles(
                        inputs=contexts[i:j],
                        prediction_length=self.prediction_length,
                        quantile_levels=q_levels,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                q = q.cpu()
                for b in range(q.shape[0]):
                    forecasts_out.append(QuantileForecast(
                        forecast_arrays=q[b].T.numpy(),
                        forecast_keys=q_keys,
                        start_date=start_dates[i + b],
                        item_id=item_ids[i + b],
                    ))
                i = j
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    continue
                raise

        return forecasts_out

class MoraiPredictor:
    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        num_samples: int = 20,
        **_,   # swallows target_dim and everything else — do NOT use it
    ) -> None:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading Moirai from %s on %s", model_path, device)

        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(model_path),
            prediction_length=prediction_length,
            context_length=2048,
            patch_size="auto",
            num_samples=num_samples,
            target_dim=1,
            past_feat_dynamic_real_dim=0,
            feat_dynamic_real_dim=0,
        )
        model.hparams.prediction_length = prediction_length
        model.hparams.target_dim = 1
        model.hparams.past_feat_dynamic_real_dim = 0

        self.predictor = model.create_predictor(batch_size=128)

    def predict(self, inputs):
        def _ensure_2d(it):
            for entry in it:
                e = dict(entry)
                for key in ("past_target", "future_target"):
                    if key in e:
                        t = np.asarray(e[key])
                        if t.ndim == 1:
                            e[key] = t[None, :]
                yield e

        return self.predictor.predict(_ensure_2d(inputs))