import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow import keras


IMG_HEIGHT = 224
IMG_WIDTH = 224


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="sum_over_batch_size", name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config


@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "best_model.keras",
        custom_objects={"FocalLoss": FocalLoss},
        compile=False
    )

    model.compile(
        optimizer="adam",
        loss=FocalLoss(),
        metrics=["accuracy"]
    )
    return model


@st.cache_data
def load_class_names():
    train_df = pd.read_csv("train_metadata.csv")
    class_names = sorted(train_df["dx"].unique().tolist())
    return class_names


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

st.title("Skin Lesion Classification App")
st.write("Upload a dermoscopic skin lesion image and get the predicted class.")

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img, img_array = preprocess_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        preds = model.predict(img_array, verbose=0)[0]
        pred_index = int(np.argmax(preds))
        pred_class = class_names[pred_index]
        confidence = float(preds[pred_index])

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {pred_class}")
        st.info(f"Confidence: {confidence:.4f}")

        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": preds
        }).sort_values(by="Probability", ascending=False)

        st.subheader("Class Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        st.bar_chart(prob_df.set_index("Class"))