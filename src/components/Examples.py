import json
import pandas as pd
import numpy as np
from .Dataloaders import ImageDataLoader, TextDataLoader, AudioDataLoader
import logging


def load_image_example():
    def load_aircraft_targets():
        with open("../assets/aircraft/images_family_train.txt") as file:
            file_list = file.read().splitlines()
        return [
            [
                f"../assets/aircraft/images/{file.split()[0]}.jpg",
                "-".join(file.split()[1:]),
            ]
            for file in file_list
        ]

    logging.info(
        "ImageDataLoader: Loading Aircraft Dataset, shuffled and limited to 500 images."
    )
    return ImageDataLoader(
        load_aircraft_targets(), name="airplane", shuffle=True, n=500
    )


def load_text_example():
    def load_tweet_classifier():
        df = pd.read_csv("./assets/tweet/Corona_NLP_train.csv", encoding="latin-1")
        df = df[["OriginalTweet", "Sentiment"]]
        return df.astype(str).values.tolist()

    logging.info(
        "TextDataLoader: Loading Aircraft Dataset, shuffled and limited to 1000 images."
    )
    return TextDataLoader(load_tweet_classifier(), name="tweets", shuffle=True, n=1000)


def load_audio_example():
    def load_esc50_classifier():
        df = pd.read_csv("./assets/esc_50/esc50.csv", encoding="latin-1")
        filenames, targets = df["filename"].tolist(), df["target"].tolist()
        return [
            [f"../assets/esc_50/audio/{filename}", target]
            for filename, target in zip(filenames, targets)
        ]

    logging.info(
        "AudioDataLoader: Loading Aircraft Dataset, shuffled and limited to 1000 images."
    )
    return AudioDataLoader(
        load_esc50_classifier(),
        name="esc_50",
        type="wav",
        normalized=False,
        shuffle=True,
        n=1000,
    )
