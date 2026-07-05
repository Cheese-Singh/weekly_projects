import speech_recognition
import pyttsx3
import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import mlx_whisper
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import time
import queue

# -------
# MODELS
# -------

recognizer = speech_recognition.Recognizer()
HF_REPO = "mlx-community/distil-whisper-large-v3"

# ----------
# CONSTANTS
# ----------

WAKE_WORDS = ["assistant", "hey assistant", "hello", "hey"]
SLEEP_WORDS = ["sleep", "end", "exit", "quit", "stop"]

TEST_FILE = "harvard.wav"

FRAMES_PER_BUFFER = 3_200
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_RATE = 16_000

# -----------------------------
# 1. SPEECH RECOGNITION BASIC
# -----------------------------

def start_speech_recognition():
    print("\n" + "=" * 60)
    print(" Running Speech Recognition - Basic ")
    print("=" * 60 + "\n")
    active = False

    while True:
        try:
            with speech_recognition.Microphone() as mic:
                print("Listening, speak now")
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                text = recognizer.recognize_google(audio).lower()
                print(f"[HEARD]: {text}")

                if any(sleep_word in text for sleep_word in SLEEP_WORDS):
                    print("Breaking chat. Goodbye!")
                    break

                if not active:
                    if any(wake_word in text for wake_word in WAKE_WORDS):
                        active = True
                        print("Activation successful")

                if active:
                    print(f"[INPUT COMMAND]: {text} \n")

        except speech_recognition.UnknownValueError:
            continue

        except speech_recognition.RequestError:
            print("API unavailable or unresponsive")
            break

    print("\n" + "=" * 60)
    print(" End of Speech Recognition ")
    print("=" * 60 + "\n")


# ------------------------------------------
# 2. USING WAVE + PLOTTING WAVE-TIME GRAPH
# ------------------------------------------

def start_wave():
    print("\n" + "=" * 60)
    print(" Running Wave Operations ")
    print("=" * 60 + "\n")

    # Reading a wave file

    if not TEST_FILE or not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"The file '{TEST_FILE}' could not be found.")
    
    print(" Reading Wave Files \n")

    try:
        with wave.open(TEST_FILE, "rb") as wav_file:
            no_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()

            print(f"Channels : {wav_file.getnchannels()}")
            print(f"Sample width : {wav_file.getsampwidth()}")
            print(f"Frame rate/Sample Rate : {frame_rate}")
            print(f"Frames : {no_frames}")
            print(f"Parameters : {wav_file.getparams()}")
            
            if frame_rate > 0:
                audio_time = no_frames/frame_rate
                print(f"Audio Time : {audio_time}")

            frames = wav_file.readframes(-1)
            print(len(frames), type(frames), type(frames[0]))
            wav_file.close()

    except wave.Error as e:
        print(f"An error occurred while parsing the WAV file format: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Writing a wave file

    print("\n Writing Wave Files \n")

    NEW_FILE = "new_harvard.wav"

    if NEW_FILE and isinstance(NEW_FILE, str):
        with wave.open(NEW_FILE, "wb") as new_wav_file:
            new_wav_file.setnchannels(2)
            new_wav_file.setsampwidth(2)
            new_wav_file.setframerate(44_100)
            new_wav_file.writeframes(frames)
            print("File successfully created\n")
    else:
        print("Invalid file name\n")

    # Plotting graphs

    if TEST_FILE:
        signal_array = np.frombuffer(frames, dtype=np.int16)
        signal_array = signal_array.reshape(-1, 2) # 2 channels
        channel1 = signal_array[:,0] # we only need 1
        times = np.linspace(0, audio_time, num=no_frames)

        plt.figure(figsize=(15,5))
        plt.plot(times, channel1)
        plt.title("Audio Signal")
        plt.ylabel("Signal Wave")
        plt.xlabel("Time")
        plt.xlim(0, audio_time)
        plt.show()

        print("Graph Plot Successful")

    print("\n" + "=" * 60)
    print(" End of Wave Operations ")
    print("=" * 60 + "\n")


# ---------------------------------
# 3. PyAudio -> Record Microphone
# ---------------------------------

def start_pyaudio():
    print("\n" + "=" * 60)
    print(" Starting PyAudio ")
    print("=" * 60 + "\n")

    p = pyaudio.PyAudio() 
    stream = None

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=FRAME_RATE,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input=True
        )

        print("Recording started")
            
        seconds = 5
        frames = []
        total_chunks = (FRAME_RATE // FRAMES_PER_BUFFER) * seconds

        for _ in range(total_chunks):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)
            
        print("Recording finished\n")
        
        recorded = wave.open("output.wav", "wb")
        recorded.setnchannels(CHANNELS)
        recorded.setframerate(FRAME_RATE)
        recorded.setsampwidth(p.get_sample_size(FORMAT))
        recorded.writeframes(b"".join(frames))
        recorded.close()

    except Exception as e:
        print(f"An unexpected error occurred {e}")
    
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

        p.terminate()
        
    print("\n" + "=" * 60)
    print(" End of PyAudio ")
    print("=" * 60 + "\n")


# -----------------------------------
# 4. AUDIO TRANSCRIPTION -> MLX Whisper
# -----------------------------------

def start_transcription(audio_file_data: np.ndarray) -> str:
    print("\n" + "=" * 60)
    print(" Starting Transcription ")
    print("=" * 60 + "\n")

    transcription = mlx_whisper.transcribe(
        audio_file_data,
        path_or_hf_repo = HF_REPO,
        language="en",
        task="transcribe",
        word_timestamps=True,
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print(" End of Transcription ")
    print("=" * 60 + "\n")
    
    return transcription["text"] if "text" in transcription else ""


# ----------------
# 5. SOUNDDEVICE 
# ----------------

def start_sounddevice(filename, duration=5, samplerate=44_100):

    print("\n" + "=" * 60)
    print(" Starting SoundDevice ")
    print("=" * 60 + "\n")

    print(f"Recording started for duration : {duration} seconds")
    audio_data = sd.rec(int(samplerate*duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio_data)
    print(f"Recording saved to {filename}")

    print("Audio Playback Now...")
    sd.play(audio_data, samplerate)
    sd.wait()

    print("\n" + "=" * 60)
    print(" SoundDevice End ")
    print("=" * 60 + "\n")

# --------------
# 6. SOUNDFILE
# --------------

def start_soundfile(filename):

    print("\n" + "=" * 60)
    print(" Starting SoundFile ")
    print("=" * 60 + "\n")

    # extracting information
    information = sf.info(filename)
    print(information)

    try:
        audio, sr = sf.read(filename, dtype='float32')
    except FileNotFoundError:
        print(f"{filename} doesn't exist")
        return
    except Exception as e:
        print(f"An error has occurred : {e}")

    print(f"Playing file : {filename}")
    sd.play(audio, sr)
    sd.wait()

    print("\n" + "=" * 60)
    print(" SoundFile End ")
    print("=" * 60 + "\n")

def adjust_volume(filename, volume_level:float):

    print("\n" + "=" * 60)
    print(" Adjusting Volume ")
    print("=" * 60 + "\n")

    print(f"Audio file : {filename}")
    print(f"Volume : {volume_level}")

    try:
        audio, sr = sf.read(filename, dtype='float32')
    except FileNotFoundError:
        print(f"{filename} doesn't exist")
        return
    except Exception as e:
        print(f"An error has occurred : {e}")

    change = volume_level * audio
    change = np.clip(change, -1.0, 1.0)
    sd.play(change, sr)
    sd.wait()

    print("\n" + "=" * 60)
    print(" Volume Adjusted")
    print("=" * 60 + "\n")

def trim_audio(filename, start_sec=0, end_sec=1):

    print("\n" + "=" * 60)
    print(" Trimming File ")
    print("=" * 60 + "\n")

    try:
        audio, sr = sf.read(filename, dtype='float32')
        frames = len(audio)
        duration = frames/sr

        start = max(0, int(start_sec * sr))
        end = min(frames, int(end_sec * sr))

        if start >= end:
            print("Invalid trim range")
            print(f"Audio duration: {duration:.2f}s")
            return

        if start == 0 and end == frames:
            print("No trimming")
            print(f"Audio duration: {duration:.2f}s")
            sd.play(audio, sr)
            sd.wait()
            return
        
        print("Audio trimmed")
        print(f"Audio trimming: {start_sec}s - {end_sec}s")
        print(f"Sample range: {start} - {end}")

        trimmed = audio[start:end]
        sd.play(trimmed, sr)
        sd.wait()
    
    except FileNotFoundError:
        print (f"{filename} not found")
    
    except Exception as e:
        print (f"An error occurred : {e}")

def voice_fade(filename, fade_duration=1.0):

    print("\n" + "=" * 60)
    print(" Voice Fading ")
    print("=" * 60 + "\n")

    try:
        audio, sr = sf.read(filename, dtype='float32')
    except FileNotFoundError:
        print(f"{filename} doesn't exist")
        return
    except Exception as e:
        print(f"An error has occurred : {e}")

    fade_len = int(fade_duration * sr)
    fade_len = min(fade_len, len(audio)//2)

    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)

    result = audio.copy()

    if audio.ndim == 1:
        result[:fade_len] *= fade_in
        result[-fade_len:] *= fade_out
    else:
        result[:fade_len, :] *= fade_in[:, None]
        result[-fade_len:, :] *= fade_out[:, None]
    
    sd.play(result, sr)
    sd.wait()

    print("\n" + "=" * 60)
    print(" Voice Fading Successful ")
    print("=" * 60 + "\n")

def normalize_audio_file(filename, target_peak = 0.9):

    print("\n" + "=" * 60)
    print(" Normalizing Audio File ")
    print("=" * 60 + "\n")

    try:
        audio, sr = sf.read(filename, dtype='float32')
    except FileNotFoundError:
        print(f"{filename} doesn't exist")
        return
    except Exception as e:
        print(f"An error has occurred : {e}")

    peak = np.max(np.abs(audio))

    if peak == 0:
        print(f"{filename} is silent. No text")
        return
    
    normalized = audio * peak / target_peak
    sd.play(normalized, sr)
    sd.wait()

    print("\n" + "=" * 60)
    print(" Audio File Normalized ")
    print("=" * 60 + "\n")

# ------------------------------
# 7. REAL TIME MICROPHONE USAGE
# ------------------------------

def start_mic(seconds = 10):

    print("\n" + "=" * 60)
    print(" Starting Mic ")
    print("=" * 60 + "\n")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)

        rms = np.sqrt(np.mean(indata ** 2))
        bars = int(rms * 100)
        print("|" * bars)

    with sd.InputStream(
        samplerate=FRAME_RATE,
        channels=CHANNELS,
        dtype='float32',
        callback=callback
    ):
        print("Listening...")
        time.sleep(seconds)
        
    print("\n" + "=" * 60)
    print(" Mic Recording Complete ")
    print("=" * 60 + "\n")

def record_until_silence(saved_audio="new.wav", silence_threshold = 0.02, silence_duration = 2.0, max_seconds = 20):
    print("\n" + "=" * 60)
    print(" Recording until silence detected ")
    print("=" * 60 + "\n")

    q = queue.Queue()
    chunks = []

    silent_time = 0
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy())
    
    try:
        with sd.InputStream(samplerate=FRAME_RATE, channels=CHANNELS, dtype='float32', callback=callback):
            print("Speak now: ")

            while True:
                chunk = q.get()
                chunks.append(chunk)

                rms = np.sqrt(np.mean(chunk ** 2))

                chunk_duration = len(chunk)/FRAME_RATE

                if rms < silence_threshold:
                    silent_time += chunk_duration
                else:
                    silent_time = 0
                
                if silent_time >= silence_duration:
                    print("Silence detected")
                    break
                
                if time.time() - start_time >= max_seconds:
                    print("Max recording time reached")
                    break

            audio = np.concatenate(chunks, axis=0)
            sf.write(saved_audio, audio, FRAME_RATE)
            print(f"File successfully saved : {saved_audio}")

    except Exception as e:
        print(f"An error has occurred : {e}")
    
    print("\n" + "=" * 60)
    print(" Recording Complete ")
    print("=" * 60 + "\n")

def detect_live_speech(seconds = 10, threshold = 0.01):
    print("\n" + "=" * 60)
    print(" Live Speech Detection ")
    print("=" * 60 + "\n")

    def callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata ** 2))

        if rms > threshold:
            print(f"Speech detected at {time.time():.3f}")
        else:
            print("Silence")
        
    with sd.InputStream(samplerate=FRAME_RATE, channels=CHANNELS, dtype='float32', callback=callback):
        time.sleep(seconds)
    
    print("\n" + "=" * 60)
    print(" Detection complete ")
    print("=" * 60 + "\n")

def large_file_streaming(filename, blocksize = 1024):
    print("\n" + "=" * 60)
    print(" Streaming By Blocks ")
    print("=" * 60 + "\n")

    try:
        with sf.SoundFile(filename) as file:
            with sd.OutputStream(
                samplerate=file.samplerate,
                channels=file.channels,
                dtype='float32'
            ) as stream:
                for block in file.blocks(blocksize=blocksize, dtype='float32'):
                    stream.write(block)
    except FileNotFoundError:
        print(f"{filename} not found")
    except Exception as e:
        print(f"An error has occurred : {e}")

    print("\n" + "=" * 60)
    print(" Streaming Complete ")
    print("=" * 60 + "\n")

# -----------
# 8. TESTING
# -----------

if __name__ == "__main__":
    start_speech_recognition()
    start_wave()
    start_pyaudio()
    print(start_transcription(TEST_FILE))
    start_sounddevice("output.wav", duration=5, samplerate=44_100)
    start_soundfile(TEST_FILE)
    adjust_volume(TEST_FILE, 0.5)
    adjust_volume(TEST_FILE, 3.0)
    trim_audio(TEST_FILE)
    trim_audio(TEST_FILE, 0, 100)
    voice_fade(TEST_FILE, 5)
    normalize_audio_file(TEST_FILE, 0.95)
    start_mic()
    record_until_silence()
    detect_live_speech()
    large_file_streaming(TEST_FILE)