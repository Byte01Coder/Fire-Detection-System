from gtts import gTTS
from playsound import playsound

def trigger_alarm(volume=50):
    alarm_message = "Warning: Fire is Detected."
    tts = gTTS(text=alarm_message, lang='en')
    tts.save("alarm1.mp3")
    playsound("alarm1.mp3")

# now we will call the function
trigger_alarm(volume=50)
