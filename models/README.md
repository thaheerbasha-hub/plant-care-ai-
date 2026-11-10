# Local model cache

The application downloads a quantized ONNX classifier, model metadata, and the author's class centroids for the out-of-scope check the first time a photo is submitted. The files are saved here and reused on later visits. They are ignored by Git.

The classifier is shared under CC BY-SA 4.0. Its model card reports evaluation on the PlantVillage collection and warns that performance on field photos can be substantially lower. This site displays raw model scores, not calibrated confidence, and does not provide treatment advice. The OOD centroids and threshold are from the model author's public scanner Space and have not been independently validated in this project.
