# Facial Emotion Recognition (FER2013)

A small deep learning project: a neural network that looks at a face image and predicts one of 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise), plus a Streamlit app where you upload a face photo and see the prediction.

**Live app:** [APP_LINK](https://emotion-recognition-fer2013.streamlit.app/)

## Dataset
[FER2013](https://www.kaggle.com/datasets/msambare/fer2013): 48×48 grayscale face images, 7 classes.
- train: 28,709 images, test: 7,178 images
- the classes are unbalanced: *happy* has 7,215 train images, *disgust* only 436

The data is **not included in this repo** (too big). Download it from Kaggle and put it like this:

```
data/
  train/  (angry, disgust, fear, happy, neutral, sad, surprise)
  test/   (same 7 folders)
```

## Project structure
```
requirements.txt        packages for the app
code/
  nn_miniproject.ipynb  data exploration, training, evaluation, error analysis
  apple.py              Streamlit app
  model2.keras          saved model used by the app
demo/                   demo of the app
data/                   (not in the repo, see above)
```

## Approach
1. **Explore:** class counts, image size and channels, sample images.
2. **Preprocess:** pixels / 255, 80/20 stratified train/validation split.
3. **Baseline:** dense network Flatten → Dense 512 → Dense 256 → Dense 7 (softmax), Adam, sparse categorical crossentropy, 30 epochs.
4. **Evaluate:** accuracy, precision, recall, confusion matrix on the test set.
5. **Error analysis:** looked at the most confident mistakes.
6. **Improve:** bigger network (1024 → 512 → 256), class weights, early stopping, and data augmentation (flip, rotation, zoom, contrast).

## Results (test set)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline dense (512-256) | 39% | 0.31 |
| Dense 1024-512-256 + class weights + early stopping (`model2`, used in the app) | 37% | 0.32 |

- Class weights helped the rare classes a little (macro recall 0.36 vs 0.33) but lowered the overall accuracy.
- Data augmentation did not help the dense network.
- Dense layers flatten the image and lose the face structure, so they stop around 40%. A CNN would be the next step.

## Run it yourself
```bash
pip install -r requirements.txt
streamlit run code/apple.py
```
Run it from the main project folder (the one that contains `code/`, `demo/` and `requirements.txt`).
To run the notebook also install: `pip install pandas matplotlib scikit-learn jupyter`

## Limitations and ethics
- **Low accuracy** (about 37-39%): wrong more than half of the time, so it must not be used for real decisions.
- **Image quality:** 48×48 grayscale only; bad lighting, angles, glasses or a hand on the face hurt it.
- **Dataset bias:** unbalanced classes (the model leans to *happy* / *neutral* and almost never says *disgust*) and noisy labels.
- **Who is in the data:** age, gender and skin tone balance are unknown, so it may work worse for some groups.
- **Expression is not feeling:** a facial expression is not always the real emotion, and it differs between people and cultures.
- **Privacy and misuse:** faces are personal data and need consent; not for surveillance, hiring or grading people.
- **The app** resizes the whole uploaded image to 48×48, so use a close-up photo of one face.
