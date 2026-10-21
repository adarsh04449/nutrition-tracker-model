# Nutrition-Tracker-Model

This project predicts four totals from one overhead food image, always in this
order: calories (kcal), protein (g), carbohydrates (g), and fat (g).

## Project structure

```text
data_pipeline/              Data acquisition and preparation
  download_data.py          Download official metadata and overhead RGB images
  prepare_data.py           Verify images and build the complete image table
  inspect_missing_ids.py    Audit images absent from official split lists
  inspect_split_sessions.py Compare timestamp sessions with official splits
  make_splits.py            Create reproducible train/validation/test tables

nutrition_model/            Model-facing code
  dataset.py                Shared preprocessing and PyTorch Dataset
  model.py                  Pretrained MobileNetV3-Large encoder check
  train.py                  Training placeholder for the next step
  evaluate.py               Evaluation placeholder
  predict.py                Prediction placeholder

data/
  raw/                      Downloaded Nutrition5k files (not committed)
  images.csv                All 3,490 verified local image/label pairs
  splits/                   Generated train, validation, and test tables
  data_report.json          Complete-data verification summary
  label_review.jpg          Human label-review sheet
```

Run commands from the project root with the virtual environment activated.

## 1. Download data

The default is a 12-image smoke test. A limit of zero downloads every available
overhead RGB image. Existing completed files are kept, so downloads are resumable.

```bash
python -m data_pipeline.download_data --limit 12
python -m data_pipeline.download_data --limit 0
```

Depth images and side-view videos are intentionally excluded.

## 2. Build and verify the complete table

```bash
python -m data_pipeline.prepare_data --visual-review passed
python -m nutrition_model.dataset --table data/images.csv
```

Preprocessing preserves aspect ratio, pads instead of cropping, converts to RGB,
and applies ImageNet normalization. Evaluation and prediction must reuse it.

## 3. Build safe splits

Nutrition5k publishes official RGB train/test ID lists. The official test set is
never sampled into training or validation. Local images absent from both official
lists are retained on disk but excluded from model development.

Validation is selected reproducibly from the official training pool. Nearby
dishes are conservatively kept together to reduce incremental-scan leakage.

```bash
python -m data_pipeline.make_splits
python -m nutrition_model.dataset --table data/splits/train.csv
python -m nutrition_model.dataset --table data/splits/validation.csv
python -m nutrition_model.dataset --table data/splits/test.csv
```

Current split counts are 2,342 training, 413 validation, and 507 official test
images. Another 228 valid local images are excluded because they do not appear in
the official RGB split lists.

## 4. Verify the pretrained encoder

```bash
python -m nutrition_model.model
```

Success reports 1,280 encoder features per image and a prediction shape of `(2, 4)`
for a two-image test batch. The four outputs are untrained normalized nutrition
predictions at this stage.

## 5. Measure the baseline

The baseline predicts the four training-set means for every validation row and
reports mean absolute error (MAE). It does not load images or inspect test labels.

```bash
python -m nutrition_model.baseline
```

Results are saved to `data/baseline_report.json`. The trained image model must be
compared on the same validation rows and should produce lower MAE.

## Next step

Train the frozen-encoder regression head. This uses AdamW, Smooth L1 loss, and
training-set target normalization. Validation MAE is reported in kcal and grams.

```bash
python -m nutrition_model.train
```

The best validation checkpoint is saved under `data/checkpoints/`. Keep the
official test set untouched until model selection is finished.

## 6. Reproduce validation results

Load the best saved checkpoint, evaluate all validation images, and compare its
MAE with the training-mean baseline:

```bash
python -m nutrition_model.evaluate
```

The result is saved to `data/validation_report.json`. This command rejects the
official test CSV during model development. After model selection, evaluate
the untouched official test split once with:

```bash
python -m nutrition_model.evaluate --split test
```

This writes `data/test_report.json`. Do not use that result to tune the model.

## Data source

Nutrition5k was released by Google Research under Creative Commons Attribution
4.0. See https://github.com/google-research-datasets/Nutrition5k for documentation.
