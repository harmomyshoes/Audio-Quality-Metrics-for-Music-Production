import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import io
from io import StringIO
from AudioMixer import AudioMixerClass
import NoiseEvalUtil as NEUtil


def UniversalAudioPlotter(
    framerate: float,
    duration: float, 
    audio_data: np.ndarray,
    title: str = "Audio Analysis",
    show_audio_player: bool = True,
    plot_layout: str = "side_by_side",  # "side_by_side", "vertical"
    color: str = "blue",
    figsize: tuple = (12, 4)
):
    """
    Universal audio plotting function that creates plots in current Streamlit context
    
    Args:
        framerate: Audio sample rate
        duration: Audio duration in seconds
        audio_data: Audio data array
        title: Title prefix for plots
        show_audio_player: Whether to show the audio player
        plot_layout: Layout style ("side_by_side", "vertical")
        color: Plot color
        figsize: Figure size tuple
    """
    
    # Add a header for this audio analysis
    st.subheader(f"📊 {title}")
        
    # Calculate time axis and frequency data
    time_axis = np.linspace(0, duration, num=len(audio_data))
    fft_spectrum = np.fft.rfft(audio_data)
    fft_freq = np.fft.rfftfreq(len(audio_data), d=1./framerate)
    fft_magnitude = np.abs(fft_spectrum)
    
    if plot_layout == "side_by_side":
        # Original side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
            fig_wave = plt.figure(figsize=(figsize[0]//2, figsize[1]))
            plt.plot(time_axis, audio_data, color=color)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"{title} - Waveform")
            plt.tight_layout()
            st.pyplot(fig_wave)
            plt.close(fig_wave)
        
        with col2:
            fig_spec = plt.figure(figsize=(figsize[0]//2, figsize[1]))
            plt.plot(fft_freq, fft_magnitude, color=color)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.title(f"{title} - Frequency Spectrum")
            plt.xlim(0, framerate/2)
            plt.tight_layout()
            st.pyplot(fig_spec)
            plt.close(fig_spec)
            
    elif plot_layout == "vertical":
        # Vertical stack layout
        fig_wave = plt.figure(figsize=figsize)
        plt.plot(time_axis, audio_data, color=color)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Waveform")
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)
        
        fig_spec = plt.figure(figsize=figsize)
        plt.plot(fft_freq, fft_magnitude, color=color)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.title(f"{title} - Frequency Spectrum")
        plt.xlim(0, framerate/2)
        plt.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)
        
    # Show audio player if requested
    if show_audio_player:
        st.audio(audio_data, sample_rate=int(framerate))


uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Load WAV file
    with wave.open(io.BytesIO(uploaded_file.read()), 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
#        print(f"Uploaded file is {n_channels}-channels.")
        sample_width = wav_file.getsampwidth()
#        print(f"Uploaded file is {sample_width}-bit.")
        if sample_width < 2:
            st.warning("Please upload a 16-bit WAV file.")
        framerate = wav_file.getframerate()
#        print(f"Uploaded file is {framerate}-Hz.")
        n_frames = wav_file.getnframes()
        # print(f"Uploaded file has {n_frames} frames.")  
        duration = n_frames / framerate
        # print(f"Uploaded file duration is {duration} seconds.")
        if duration > 8:
            st.warning("Audio is longer than 8 seconds and will be trimmed to the first 8 seconds.")
        audio_data = wav_file.readframes(n_frames)
        ReferAudio = AudioMixerClass(audio_data, n_channels, sample_width, framerate, n_frames, duration)
        st.session_state.mixer = ReferAudio
        Refer_Mp3_data = ReferAudio.GeneratingMP3Ref(ReferAudio.InitalData, bitrate=64)
        st.session_state.Refer_Mp3_data = Refer_Mp3_data
        # st.warning(f"Refer_Mp3_data shape is {Refer_Mp3_data.shape}, framerate is {ReferAudio.SampleRate}, duration is {ReferAudio.Duration}")
    if 'Refer_Mp3_data' and 'mixer' in st.session_state:
        # PaintWithAudio(st.session_state.mixer.SampleRate,st.session_state.mixer.Duration,st.session_state.Refer_Mp3_data)
        UniversalAudioPlotter(
            framerate=st.session_state.mixer.SampleRate,
            duration=st.session_state.mixer.Duration,
            audio_data=st.session_state.Refer_Mp3_data,
            title="Original Audio",
            color="blue",
            plot_layout="vertical"  # Try different layout
        )
    else:
        st.warning("No MP3 data available for playback.")

st.title("🔧 Adding Degradation Effects")

# Sliders with distinct default values and ranges
SigtoHum_value = st.slider("🟣 Signal-to-Hum Noise Level (dB)", min_value=1.0, max_value=80.0, value=60.0)
SigtoHiss_value = st.slider("🟢 Signal-to-Hiss Noise Level (dB)", min_value=1.0, max_value=80.0, value=60.0)
ClipPercentage_value = st.slider("🔴 Percentage of Audio to Clip (%)", min_value=0.0, max_value=80.0, value=3.0)
DropOccurence_value = st.slider("🟠 Number of Glitch (Dropout) Occurrences", min_value=0, max_value=80, value=0)


# Rephrased plain English summary
st.markdown(
    f"""
**You have applied the following degradations to the audio:**

- <span style='color:#6C3483;'>Signal-to-Hum noise ratio</span>: <b>{SigtoHum_value} dB</b>
- <span style='color:#229954;'>Signal-to-Hiss noise ratio</span>: <b>{SigtoHiss_value} dB</b>
- <span style='color:#CB4335;'>Clipping</span>: <b>{ClipPercentage_value}%</b> of the audio will be clipped
- <span style='color:#CA6F1E;'>Glitches</span>: <b>{DropOccurence_value}</b> dropouts will be inserted

""",
    unsafe_allow_html=True
)


# Button triggers the evaluation and passes uploaded_file directly
if st.button("Applied The Degradation"):
    if uploaded_file is not None:
        # Just pass uploaded_file to your function!
        degraded_audio, new_sr= st.session_state.mixer.TestNoisedOnlyFile([SigtoHum_value,SigtoHiss_value,ClipPercentage_value,DropOccurence_value])
        Degraded_Mp3_data = st.session_state.mixer.GeneratingMP3Ref(degraded_audio, bitrate=64)
        st.session_state.degraded_audio = Degraded_Mp3_data.flatten()
        st.session_state.degraded_sr    = new_sr
        st.success("✅ Degradation evaluation applied!")
        # st.warning(f"degraded_Mp3_data shape is {st.session_state.degraded_audio.shape}, framerate is {st.session_state.degraded_sr}, duration is {ReferAudio.Duration}")
        # You can now play, plot, or allow download of degraded_audio
        if 'degraded_audio' and 'mixer' in st.session_state:
            # PaintWithAudio(st.session_state.mixer.SampleRate,st.session_state.mixer.Duration,st.session_state.degraded_audio)
            UniversalAudioPlotter(
                framerate=st.session_state.mixer.SampleRate,
                duration=st.session_state.mixer.Duration,
                audio_data=st.session_state.degraded_audio,
                title="Degraded Audio",
                color="red",
                plot_layout="vertical"  # Try different layout
            )
        else:
            st.warning("No degraded audio available for playback.")
    else:
        st.error("⚠️ Please upload a WAV file before running evaluation.")

st.title("🚦 Degradation Level Evaluation")

if st.button("HAAQI Evaluation"):
    if 'mixer' not in st.session_state or st.session_state.mixer.SampleRate != 48000:
        st.error("⚠️ HAAQI evaluation requires a 48kHz audio file.")

    if uploaded_file is not None and 'degraded_audio' in st.session_state and 'mixer' in st.session_state:
        # Initialise HAAQI evaluation
        import clarity.evaluator.haaqi as haaqi
        from clarity.utils.audiogram import Audiogram
        levels_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        audiogram_NH = Audiogram(levels=levels_1)

        srate = st.session_state.mixer.SampleRate
        HAAQI_score = round(haaqi.compute_haaqi(st.session_state.degraded_audio,st.session_state.Refer_Mp3_data,srate,srate,audiogram_NH),2)
        st.success(f"HAAQI Score: **{HAAQI_score}**")

        # 4) Show static explainer
        with st.expander("ℹ️ What is HAAQI and how do I interpret the score?"):
            st.markdown(
                """
                **HAAQI** (Hearing-Aid Audio Quality Index) predicts perceived audio fidelity for hearing-aid users.  
                It has been shown to reliably quantify the impact of various degradations—including those typical in music-production pipelines—against a clean reference.

                For full details and the underlying methodology, see the [ Technical Specification (PDF)](DAFx25_paper_5_Evaluation_with_HAAQI.pdf).

                **Score Interpretation**
                - **1.00**: Perfect match to the reference (indistinguishable)  
                - **0.80–1.00**: Generally acceptable quality  
                - **0.60–0.80**: Distortions or noise become perceptible  
                - **< 0.50**: Severe degradation, rendering audio essentially unusable  
                """
            )

        # 5) Dynamic interpretation (optional)
        if HAAQI_score >= 0.8:
            st.info("✅ This version is within **acceptable** quality range.")
        elif HAAQI_score >= 0.6:
            st.warning("⚠️ Artifacts are **perceptible**; consider adjusting your degradation settings.")
        else:
            st.error("❌ Quality is **severely degraded**; you’ll likely need to reprocess or reduce noise levels.")
    else:
        st.error("⚠️ Please ensure you have uploaded a WAV file and applied degradation before running HAAQI evaluation.")    
