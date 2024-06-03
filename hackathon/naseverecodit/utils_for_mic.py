import speech_recognition as sr
import os, time

recognizer = sr.Recognizer()
microphone = sr.Microphone()

recognized_data = ""

def record_mic_audio(is_recording):
    global recognized_data

    print("Starting recording...")

    if os.path.exists("audio/microphone-results.wav"):
        os.remove("audio/microphone-results.wav")

    with microphone as source:

        if is_recording:
            print("Listening...")
            audio = recognizer.listen(source, timeout=None)
            with open("audio/microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

def audio_recognition():
    global recognized_data
    audio_file_path = "audio/microphone-results.wav"

    while not os.path.exists(audio_file_path):
        time.sleep(0.1)

    myaudio = sr.AudioFile(audio_file_path)
    with myaudio as source:
        audio = recognizer.record(source)
        try:
            recognized_data = recognizer.recognize_google(audio, language="ru")
            return recognized_data
        except sr.exceptions.UnknownValueError:
            return "Не предвиденная ошибка, попробуйте снова записать микрофон"

