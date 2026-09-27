# models

Trained model weights and edge build artifacts are not stored in this repository. They are build output rather than source, and can be regenerated from the training scripts and the datasets.

Place local checkpoints and exported models (for example TFLite or ONNX) under this folder. Nothing in this folder except this README is tracked by git.

1. Download the MRL Eye dataset from [mrl.cs.vsb.cz/eyedataset.html](https://mrl.cs.vsb.cz/eyedataset.html) (free) and extract it into `data/mrlEyes_2018_01/`.
2. Run:

   ```bash
   uv run python -m drivesafe.models.train_eye_state_cnn

