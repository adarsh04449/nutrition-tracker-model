# Verification completed

- Loaded pretrained MobileNetV3-Large and verified feature shape `(1, 1280)`.
- Parsed 5,006 unique official metadata records.
- Downloaded and decoded all 3,490 available overhead RGB images (1.4 GiB).
- Built 3,490 unique image/label rows with four finite, nonnegative targets.
- Confirmed table paths exactly cover all downloaded RGB files.
- Completed human review with no obvious photo/label mismatches.
- Downloaded the official RGB train/test ID lists.
- Investigated 228 valid local dishes absent from both official lists; they are excluded.
- Created deterministic, group-aware splits with seed 42:
  - train: 2,342
  - validation: 413
  - official test: 507
- Verified zero overlap and loaded every split through the PyTorch data pipeline.
- Measured the training-mean baseline on 413 validation dishes:
  - calories MAE: 180.38 kcal
  - protein MAE: 18.10 g
  - carbohydrates MAE: 12.03 g
  - fat MAE: 10.32 g
- Built the frozen MobileNetV3-Large regression model:
  - 1,280 encoder features per image
  - four normalized nutrition outputs
  - 5,124 trainable regression-head parameters
  - verified gradients do not update the frozen encoder
- Smoke-tested the complete training pipeline on two train and validation batches:
  - training-only target mean and standard-deviation calculation
  - AdamW optimization with Smooth L1 loss
  - denormalized validation MAE in kcal and grams
  - best-checkpoint and training-history saving
- Completed 10 epochs of frozen-encoder head training; epoch 9 was retained.
- Reproduced the best checkpoint on all 413 validation images:
  - calories MAE: 92.11 kcal (48.94% below baseline)
  - protein MAE: 9.97 g (44.93% below baseline)
  - carbohydrates MAE: 7.84 g (34.84% below baseline)
  - fat MAE: 6.59 g (36.19% below baseline)
- Confirmed the development evaluator rejects the official test CSV.
- Evaluated the epoch-9 frozen-encoder checkpoint once on the untouched official
  test split (507 images); results are saved in `data/test_report.json`:
  - calories MAE: 73.64 kcal (56.17% below the training-mean baseline)
  - protein MAE: 7.67 g (48.84% below the training-mean baseline)
  - carbohydrates MAE: 8.06 g (41.24% below the training-mean baseline)
  - fat MAE: 6.05 g (43.08% below the training-mean baseline)

Independent real-world accuracy has not been measured yet. The official test
set is now considered consumed for this model version and must not be used for
further model tuning.
