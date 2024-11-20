import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import torchaudio

# Load your fine-tuned model and processor
MODEL_PATH = "./wav2vec2-large-xlsr-lingala2"
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH, use_safetensors=True)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# Function to process audio and transcribe
def transcribe_audio(audio_data, sample_rate=16000):
    # Convert audio data to tensor
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Preprocess audio
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predicted IDs to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

# Audio processing class for WebRTC
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.sample_rate = 16000

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.audio_frames.extend(audio.tolist())  # Append audio data
        return av.AudioFrame.from_ndarray(audio, format="s16").reformat(layout="mono", rate=16000)

# Streamlit app
st.title("Lingala ASR with Live Audio Recording")
st.write("Record yourself speaking Lingala and get real-time transcriptions.")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
)

# Transcription Section
if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("Recording in progress... Please speak.")
    
    if webrtc_ctx.audio_processor:
        # Collect audio frames
        processor = webrtc_ctx.audio_processor
        if len(processor.audio_frames) > 0:
            st.write("Transcribing...")
            audio_data = np.array(processor.audio_frames, dtype=np.float32)
            
            # Transcribe audio
            transcription = transcribe_audio(audio_data, sample_rate=16000)
            st.write(f"**Transcription:** {transcription}")
