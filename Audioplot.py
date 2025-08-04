import streamlit as st

from audio_plot import audio_plot, Events

# minimal example
audio_plot(
    embeddings = [[0.1, 0.2], [0.3, 0.4]],
    urls = [
        "https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3#t=0.0,3.0",
        "https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3#t=60.0,63.0",
    ]
)

# e.g. metadata is a pd.DataFrame
audio_plot(
    embeddings=metadata[["valence", "arousal"]],
    urls=metadata["urls"],
    labels=[f"{artist}: {title}" for artist, title in zip(metadata["artist"], metadata["title"])],
    event=Events.CLICK,  # or Events.HOVER
    height=800,
    volume=0.4,  # between 0. and 1.
    hidePlayer=true,
)