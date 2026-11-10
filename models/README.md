# Optional model files

The website's disease-classification panel stays disabled until a compatible model is supplied.

- `plantcare_effnet_b0.pth`: EfficientNet-B0 state dictionary, with the final classifier layer sized to the number of classes.
- `class_labels.json`: JSON list of class names in the same order used during training.

The `.pth` file is intentionally ignored by Git. Before sharing or deploying a model, verify its training data, evaluation results, and redistribution terms. Do not treat raw model scores as calibrated confidence unless calibration has been fitted and documented.
