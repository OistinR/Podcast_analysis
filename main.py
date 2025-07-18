import whisper
from pyannote.audio import Pipeline
from yt_dlp import YoutubeDL
import os
import csv

HF_TOKEN = os.getenv("HF_TOKEN")

def download_youtube_audio(url, output_file='audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded.%(ext)s',
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    os.rename('downloaded.wav', output_file)
    return output_file


def transcribe_whisper(audio_path):
    model = whisper.load_model("base").to("cuda")   # or medium/large
    print("Model device:", next(model.parameters()).device)  # Confirm it's on CUDA
    result = model.transcribe(audio_path)
    return result['segments']

def run_diarization(audio_path, hf_token):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })
    return segments


def align_speakers(transcript, diarization):
    merged = []
    for t in transcript:
        for d in diarization:
            if d['start'] <= t['start'] <= d['end']:
                merged.append({
                    'start': t['start'],
                    'end': t['end'],
                    'text': t['text'],
                    'speaker': d['speaker']
                })
                break
    return merged


if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=D_sPkkJuGMI" # TLDR news channel discussing Russia's Economy
    hf_token = HF_TOKEN

    # audio_path = download_youtube_audio(youtube_url)
    # print("--------Download Comnplete--------------")
    audio_path = "audio.wav"
    transcript = transcribe_whisper(audio_path)
    print("--------Transcription Comnplete--------------")
    diarization = run_diarization(audio_path, hf_token)
    print("--------Diarization Comnplete--------------")
    final_output = align_speakers(transcript, diarization)
    
    # for entry in final_output:
    #     print(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['speaker']}: {entry['text']}")

    with open("transcript_output.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["start", "end", "speaker", "text"])
        writer.writeheader()
        writer.writerows(final_output)
    
    print("--------File Writting Comnplete--------------")