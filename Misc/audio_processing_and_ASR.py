import speech_recognition
import pyttsx3
import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import mlx_whisper

# -------
# MODELS
# -------

recognizer = speech_recognition.Recognizer()
HF_REPO = "mlx-community/distil-whisper-large-v3"

# ----------
# CONSTANTS
# ----------

WAKE_WORDS = ["assistant", "hey assistant", "hello", "hey"]
SLEEP_WORDS = ["sleep", "end", "exit", "quit"]

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
# AUDIO TRANSCRIPTION -> MLX Whisper
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

                   
if __name__ == "__main__":
    start_speech_recognition()
    start_wave()
    start_pyaudio()
    print(start_transcription("harvard.wav"))