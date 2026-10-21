# Nutrition5k Nutrition-Estimation Project Plan

This file records the project status and the next planned experiments.

## Current status

- [x] Download and verify the overhead RGB Nutrition5k images.
- [x] Build the complete image/label table.
- [x] Create leakage-aware train, validation, and official test splits.
- [x] Measure the training-mean baseline.
- [x] Build the MobileNetV3-Large nutrition regression model.
- [x] Train a frozen-encoder regression head.
- [x] Select the best frozen-head checkpoint using validation performance.
- [x] Evaluate the frozen-head checkpoint on the official test set.

The current frozen-head checkpoint is `data/checkpoints/best_head.pt`.

Official test results for this checkpoint are saved in
`data/test_report.json`. The model achieved these mean absolute errors:

- Calories: 73.64 kcal
- Protein: 7.67 g
- Carbohydrates: 8.06 g
- Fat: 6.05 g

The official test set should not be used while tuning future models. It is a
benchmark for the current model version.

## Next phase: controlled fine-tuning

The current model trained only the regression head while keeping the
MobileNetV3-Large encoder frozen. Fine-tuning will allow the final encoder
layers to adapt to food-specific visual patterns.

- [ ] Add a controlled fine-tuning mode to `nutrition_model/train.py`.
- [ ] Load the current best frozen-head checkpoint before fine-tuning.
- [ ] Unfreeze only the final encoder block or blocks initially.
- [ ] Use a small encoder learning rate, such as `1e-5`.
- [ ] Keep a larger learning rate for the regression head, such as `1e-3`.
- [ ] Monitor validation loss and validation MAE after every epoch.
- [ ] Use early stopping or retain only the best validation checkpoint.
- [ ] Compare the fine-tuned model against the frozen-head model on validation.

The fine-tuned model is kept only if it improves validation performance without
unstable training or obvious overfitting.

## Final model evaluation

- [ ] Choose the final model using validation results only.
- [ ] Evaluate the selected final model on the official test set once.
- [ ] Save a separate final test report.
- [ ] Record the model configuration and metrics in `VERIFIED.md`.

If the fine-tuned model is not better, keep the frozen-head checkpoint as the
final model for this version.

## Prediction workflow

- [x] Complete `nutrition_model/predict.py`.
- [ ] Load the selected checkpoint and its training target statistics.
- [ ] Reuse the exact training preprocessing pipeline.
- [ ] Accept an image path from the command line.
- [ ] Print calories, protein, carbohydrates, and fat in real units.
- [x] Test prediction mode on known Nutrition5k olive images.

Example target command:

```bash
python -m nutrition_model.predict --image path/to/food.jpg
```

## Portion-size experiment

This experiment tests whether predictions increase when the amount of the same
food increases.

- [ ] Choose one food, such as cooked rice.
- [ ] Prepare measured portions such as 100 g, 200 g, and 300 g.
- [ ] Photograph each portion with the same plate, camera, distance, lighting,
  and background.
- [ ] Run each image through `predict.py`.
- [ ] Check whether predicted nutrients increase monotonically.
- [ ] Compare predictions with reference nutrition values based on measured
  weight.

This tests portion sensitivity. It is not a replacement for a larger
real-world accuracy evaluation.

## Real-world evaluation

- [ ] Collect approximately 50–100 food photos outside Nutrition5k.
- [ ] Record measured portions or reliable reference nutrition values.
- [ ] Run the final model on those images.
- [ ] Calculate MAE for calories, protein, carbohydrates, and fat.
- [ ] Record failure cases such as unusual foods, poor lighting, occlusion,
  changing camera distance, and different plate sizes.

Unlabeled photographs can test whether the prediction script works, but they
cannot measure nutrition accuracy.

## Rules for evaluation

- Target order is always:

  ```text
  calories_kcal, protein_g, carbohydrates_g, fat_g
  ```

- Target normalization statistics must be calculated from the training set
  only.
- Validation data is used for model selection and tuning.
- The official test data is reserved for final benchmarking.
- Real-world accuracy requires images with reference nutrition values.

## Experiment log: existing Nutrition5k portion pairs

These are preliminary portion-sensitivity tests using single-ingredient dishes
with measured metadata weights. The model used the epoch-9 frozen-head
checkpoint. Predictions were run with `nutrition_model.predict`.

### Olives

- Smaller: `dish_1556572657`, 36 g, 41.4 kcal label.
- Larger: `dish_1556575014`, 62 g, 71.3 kcal label.
- Predictions: 99.4 kcal -> 132.8 kcal; 1.37 g -> 4.56 g protein;
  6.05 g -> 5.49 g carbohydrates; 9.37 g -> 10.10 g fat.
- Result: calories, protein, and fat increased; carbohydrates decreased.

### White rice

- Smaller: `dish_1558458737`, 32 g, 41.6 kcal label.
- Larger: `dish_1558459276`, 79 g, 102.7 kcal label.
- Predictions: 58.0 kcal -> 112.0 kcal.
- Result: calorie, protein, and carbohydrate predictions increased; fat
  decreased slightly. This was the strongest positive portion response.

### Brussels sprouts

- Smaller: `dish_1559675101`, 31 g, 13.3 kcal label.
- Larger: `dish_1558724574`, 118 g, 50.7 kcal label.
- Predictions: 82.5 kcal -> 51.7 kcal.
- Result: the calorie prediction moved in the wrong direction. This pair shows
  that portion sensitivity is not reliable yet.

### Broccoli

- Smaller: `dish_1559841950`, 32 g, 11.2 kcal label.
- Larger: `dish_1558724723`, 112 g, 39.2 kcal label.
- Predictions: 54.9 kcal -> 96.9 kcal.
- Result: calories increased, but protein and carbohydrates decreased. Plate
  presentation differed more, so this is a weaker comparison.

### Findings from these tests

- The model sometimes responds to larger portions, especially for calories.
- The response is not consistently monotonic across foods or nutrients.
- Predictions can be negative for some nutrients because the current linear
  regression head has no nonnegative-output constraint.
- These are useful preliminary tests, not a replacement for a controlled
  measured real-world evaluation.

## Future tests and improvements

- [ ] Test more same-ingredient pairs using metadata weights.
- [ ] Calculate a portion monotonicity score for each nutrient across many
  ordered pairs.
- [ ] Compare predicted nutrient ratios with measured-weight ratios.
- [ ] Create a controlled 100 g / 200 g / 300 g photo experiment using the same
  food, plate, camera, and lighting.
- [ ] Investigate nonnegative output methods, such as a suitable output
  activation or target transformation, and compare them on validation data.
- [ ] Fine-tune the final MobileNetV3 encoder layers using validation-based
  model selection.
- [ ] Check whether fine-tuning reduces negative predictions and improves
  portion monotonicity.
- [ ] Test robustness to camera distance, lighting, plate type, food position,
  and background.
- [ ] Collect 50–100 external food images with measured portions and reference
  nutrition values.
- [ ] Report real-world MAE, monotonicity, calibration, and failure cases.
