from IPython.display import display, Javascript, Audio
from google.colab.output import eval_js
from base64 import b64decode
from google.cloud import texttospeech
import io
import os
from PIL import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set the path to your Google Cloud service account key file
key_file_path = '/content/instant-carrier-408618-364814e5eef8.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path

# Initialize the Text-to-Speech client
client = texttospeech.TextToSpeechClient()

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# JavaScript to handle camera display and voice-activated image capture
js = Javascript('''
async function showCamera() {
    const div = document.createElement('div');
    const startBtn = document.createElement('button');
    startBtn.textContent = 'Start Voice Command';
    div.appendChild(startBtn);

    const video = document.createElement('video');
    video.style.display = 'block';
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });

    document.body.appendChild(div);
    div.appendChild(video);
    video.srcObject = stream;
    await video.play();

    google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

    return new Promise((resolve) => {
        startBtn.onclick = () => {
            resolve(takePhoto());
        };
    });
}

async function takePhoto() {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    const video = document.querySelector('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    recognition.start();
    return new Promise((resolve) => {
        recognition.onresult = (event) => {
            const speechToText = event.results[0][0].transcript;
            if (speechToText.toLowerCase() === 'click') {
                canvas.getContext('2d').drawImage(video, 0, 0);
                video.srcObject.getVideoTracks()[0].stop();
                video.remove();
                recognition.stop();
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            }
        };

        recognition.onerror = (event) => {
            video.srcObject.getVideoTracks()[0].stop();
            video.remove();
            recognition.stop();
            resolve('');
        };
    });
}

''')

# Display the JavaScript code
display(js)

# Function to capture and display the image
def take_photo():
    data = eval_js('showCamera()')
    if data == '':
        print("No photo captured.")
        return None
    binary = b64decode(data.split(',')[1])
    return PILImage.open(io.BytesIO(binary))

# Function to process and display the image with caption
def process_and_display_image(image):
    try:
        # Display the image
        display(image)

        # Generate caption using BLIP
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Predefined caption
        print("Generated Caption:", caption)

        # Synthesize speech from the caption
        synthesis_input = texttospeech.SynthesisInput(text=caption)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-D", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        # Play the audio
        display(Audio(response.audio_content, autoplay=True))
        print("Caption:", caption)

    except Exception as e:
        print("Error:", e)

# Main execution: Capture the image and then process it
captured_image = take_photo()
if captured_image:
    process_and_display_image(captured_image)

