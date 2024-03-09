import torch
import gradio as gr
from transformers import pipeline


MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return text



demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    title="Whisper Large V3: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath", optional=True, label="Audio file"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V3: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
    ),
    allow_flagging="never",
)



with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe], ["Microphone", "Audio file"])

demo.launch()

