import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from groq import Groq
import time

# Load environment variables
load_dotenv()

# Initialize the Hugging Face InferenceClient and Groq client
client = InferenceClient()
client2 = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit application
def main():
    st.title("VoiceVoyager")
    st.caption("The linguistic chameleon that clones your voice and speaks Hindi, so you don't have to learn a new language to go viral in Varanasi.")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner('Processing...'):
            # Save the uploaded audio temporarily
            input_file_path = "input_audio.wav"
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Perform automatic speech recognition
            response = client.automatic_speech_recognition(input_file_path)
            text_eng = response.text

            # Display the recognized text
            st.subheader("Recognized English Text")
            st.write(text_eng)

            # Create a loading message
            with st.spinner('Translating to Hinglish...'):
                time.sleep(2)  # Simulate a delay for loading message

                # Generate Hinglish translation
                completion = client2.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert English to Hinglish Translator. The translated text should sound natural and also convert all the difficult words and phrases in English to Hinglish. The translated text must be able to keep certain words in English to keep the Hindi translation Easy. ### Example: English: I had about a 30 minute demo just using this new headset Hinglish: मुझे सिर्फ ३० minute का demo मिला था नये headset का इस्तमाल करने के लिए ### Generate a dataset of 5 examples for English to Hinglish translation where Hindi words should be in Devanagari and English words should be in English. Use the above example as a reference. Create examples biased towards content creators."
                        },
                        {
                            "role": "user",
                            "content": "English:" + text_eng
                        }
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )

                response_string = ""
                for chunk in completion:
                    response_string += chunk.choices[0].delta.content or ""

                # Display the translated text
                st.subheader("Translated Hinglish Text")
                st.write(response_string)

                # Convert translated text to speech
                final_resp = client.text_to_speech(response_string, model="facebook/mms-tts-hin")

                # Save the translated speech temporarily
                output_file_path = "translated_speech.wav"
                with open(output_file_path, "wb") as f:
                    f.write(final_resp)

                # Play the translated audio
                st.audio(output_file_path, format="audio/wav")
                st.success("Translation and speech synthesis completed!")

if __name__ == "__main__":
    main()
