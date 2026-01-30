## Models de CNN per classificació d’imatges
Els models i documents complementaris estan pujats a Hugging Face:
[https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier)
# CNN Models for Image Classification

Overview
- End-to-end CNN pipeline using TensorFlow/Keras.
- Designed for small/medium datasets (few thousands of images).

How data must be structured
data/
 ├── train/
 │   ├── classA/
 │   │   ├── img001.jpg
 │   │   └── ...
 │   └── classB/
 └── val/
     └── ...

Run training
1. pip install -r requirements.txt
2. python src/train.py --data_dir data --epochs 20 --batch_size 32 --save_path models/cnn_model.h5

Inference
python src/predict.py --model models/cnn_model.h5 --image_path samples/img1.jpg
