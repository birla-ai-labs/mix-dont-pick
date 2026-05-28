# Changes from upstream uni2ts (base: main branch, ~v2.0.0)

## Bug fix
- `cli/train.py`: handle `logger.version = None` in seed computation
  (line 154: `cfg.seed + (trainer.logger.version or 0)`)

## Training configuration
- `cli/conf/pretrain/default.yaml`: switched to WandB logger, bf16-mixed 
  precision, batch size 96, 50 epochs × 1000 steps/epoch
- `cli/conf/pretrain/model/moirai_small.yaml`: warmup steps 10k -> 1k

## New: mixed corpus support  
- `src/uni2ts/data/builder/concat_simple.py`: MixedDatasetBuilder class
  that concatenates multiple SimpleDatasetBuilder corpora via ConcatDataset.
  Required for mixed_all11 and real–synthetic mixture conditions.

## New: per-condition data configs
- `cli/conf/pretrain/data/*.yaml`: one Hydra config per training condition
  (rename fringe_ets -> ets, center_waveform -> waveform, baseline_fbm -> fbm
  before release)