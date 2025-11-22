import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import random
import io
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="SoulSync-Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1DB954;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .emotion-button {
        font-size: 1.2em;
        padding: 0.5em 1em;
        margin: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

# Music Database - Curated songs for each emotion
MUSIC_DATABASE = {
    "happy": [
        {"song": "Happy", "artist": "Pharrell Williams", "language": "English"},
        {"song": "Can't Stop The Feeling", "artist": "Justin Timberlake", "language": "English"},
        {"song": "Walking on Sunshine", "artist": "Katrina and The Waves", "language": "English"},
        {"song": "Good Vibrations", "artist": "The Beach Boys", "language": "English"},
        {"song": "Don't Stop Me Now", "artist": "Queen", "language": "English"},
        {"song": "Uptown Funk", "artist": "Mark Ronson ft Bruno Mars", "language": "English"},
        {"song": "I Gotta Feeling", "artist": "Black Eyed Peas", "language": "English"},
        {"song": "Shake It Off", "artist": "Taylor Swift", "language": "English"},
        {"song": "Badtameez Dil", "artist": "Benny Dayal", "language": "Hindi"},
        {"song": "Gallan Goodiyaan", "artist": "Yashita Sharma", "language": "Hindi"},
        {"song": "Kala Chashma", "artist": "Amar Arshi", "language": "Hindi"},
        {"song": "Dil Dhadakne Do", "artist": "Priyanka Chopra", "language": "Hindi"},
        {"song": "Balam Pichkari", "artist": "Shalmali Kholgade", "language": "Hindi"},
        {"song": "London Thumakda", "artist": "Labh Janjua", "language": "Hindi"},
    ],
    "sad": [
        {"song": "Someone Like You", "artist": "Adele", "language": "English"},
        {"song": "Fix You", "artist": "Coldplay", "language": "English"},
        {"song": "The Night We Met", "artist": "Lord Huron", "language": "English"},
        {"song": "All I Want", "artist": "Kodaline", "language": "English"},
        {"song": "Hurt", "artist": "Johnny Cash", "language": "English"},
        {"song": "Mad World", "artist": "Gary Jules", "language": "English"},
        {"song": "Tears in Heaven", "artist": "Eric Clapton", "language": "English"},
        {"song": "Say Something", "artist": "A Great Big World", "language": "English"},
        {"song": "Tum Hi Ho", "artist": "Arijit Singh", "language": "Hindi"},
        {"song": "Channa Mereya", "artist": "Arijit Singh", "language": "Hindi"},
        {"song": "Agar Tum Saath Ho", "artist": "Alka Yagnik", "language": "Hindi"},
        {"song": "Kabira", "artist": "Tochi Raina", "language": "Hindi"},
        {"song": "Ae Dil Hai Mushkil", "artist": "Arijit Singh", "language": "Hindi"},
        {"song": "Phir Bhi Tumko Chaahunga", "artist": "Arijit Singh", "language": "Hindi"},
    ],
    "angry": [
        {"song": "Break Stuff", "artist": "Limp Bizkit", "language": "English"},
        {"song": "In The End", "artist": "Linkin Park", "language": "English"},
        {"song": "Killing in the Name", "artist": "Rage Against The Machine", "language": "English"},
        {"song": "Blow Me Away", "artist": "Breaking Benjamin", "language": "English"},
        {"song": "The Way I Am", "artist": "Eminem", "language": "English"},
        {"song": "You're Gonna Go Far Kid", "artist": "The Offspring", "language": "English"},
        {"song": "Bodies", "artist": "Drowning Pool", "language": "English"},
        {"song": "Numb", "artist": "Linkin Park", "language": "English"},
        {"song": "Apna Time Aayega", "artist": "Ranveer Singh", "language": "Hindi"},
        {"song": "Sultan", "artist": "Sukhwinder Singh", "language": "Hindi"},
        {"song": "Dangal", "artist": "Daler Mehndi", "language": "Hindi"},
        {"song": "Zinda", "artist": "Siddharth Mahadevan", "language": "Hindi"},
        {"song": "Malhari", "artist": "Vishal Dadlani", "language": "Hindi"},
    ],
    "surprise": [
        {"song": "Thunderstruck", "artist": "AC/DC", "language": "English"},
        {"song": "Eye of the Tiger", "artist": "Survivor", "language": "English"},
        {"song": "Bohemian Rhapsody", "artist": "Queen", "language": "English"},
        {"song": "Mr. Blue Sky", "artist": "Electric Light Orchestra", "language": "English"},
        {"song": "September", "artist": "Earth Wind and Fire", "language": "English"},
        {"song": "Crazy", "artist": "Gnarls Barkley", "language": "English"},
        {"song": "Pump It", "artist": "Black Eyed Peas", "language": "English"},
        {"song": "Starman", "artist": "David Bowie", "language": "English"},
        {"song": "Ghungroo", "artist": "Arijit Singh", "language": "Hindi"},
        {"song": "Dil Chahta Hai", "artist": "Shankar Mahadevan", "language": "Hindi"},
        {"song": "Nacho Nacho", "artist": "Vishal Mishra", "language": "Hindi"},
        {"song": "Senorita", "artist": "Farhan Akhtar", "language": "Hindi"},
        {"song": "Nashe Si Chadh Gayi", "artist": "Arijit Singh", "language": "Hindi"},
    ],
    "uplifting": [
        {"song": "Eye of the Tiger", "artist": "Survivor", "language": "English"},
        {"song": "Don't Stop Believin", "artist": "Journey", "language": "English"},
        {"song": "Here Comes The Sun", "artist": "The Beatles", "language": "English"},
        {"song": "Beautiful Day", "artist": "U2", "language": "English"},
        {"song": "Three Little Birds", "artist": "Bob Marley", "language": "English"},
        {"song": "Lovely Day", "artist": "Bill Withers", "language": "English"},
        {"song": "Brave", "artist": "Sara Bareilles", "language": "English"},
        {"song": "Roar", "artist": "Katy Perry", "language": "English"},
        {"song": "Zindagi Na Milegi Dobara", "artist": "Shankar Mahadevan", "language": "Hindi"},
        {"song": "Kar Har Maidaan Fateh", "artist": "Sukhwinder Singh", "language": "Hindi"},
        {"song": "Chak De India", "artist": "Sukhwinder Singh", "language": "Hindi"},
        {"song": "Lakshya", "artist": "Shankar Mahadevan", "language": "Hindi"},
        {"song": "Jashn-e-Bahara", "artist": "Javed Ali", "language": "Hindi"},
    ]
}

def analyze_emotion_from_text(text):
    """
    Analyze emotion from text using sentiment analysis
    Returns: emotion category (happy, sad, angry, surprise)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Check for specific emotion keywords
    text_lower = text.lower()
    
    # Anger detection
    anger_keywords = ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'hate', 'rage']
    if any(keyword in text_lower for keyword in anger_keywords):
        return "angry"
    
    # Surprise detection
    surprise_keywords = ['surprise', 'shocked', 'amazed', 'astonished', 'wow', 'unexpected']
    if any(keyword in text_lower for keyword in surprise_keywords):
        return "surprise"
    
    # Sentiment-based detection
    if polarity > 0.3:
        return "happy"
    elif polarity < -0.3:
        return "sad"
    elif subjectivity > 0.6 and polarity > 0:
        return "surprise"
    elif polarity < 0:
        return "angry"
    else:
        return "happy"  # Default to happy for neutral

def recognize_speech_from_audio(audio_bytes):
    """
    Recognize speech from audio bytes (WAV format)
    Returns: recognized text or None
    """
    recognizer = sr.Recognizer()
    
    try:
        # Convert audio bytes to a file-like object and use AudioFile
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio. Please speak clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        return None

def recommend_music(emotion, uplifting_mode=False, language="Any"):
    """
    Recommend music based on emotion and language
    Returns: song dictionary
    """
    if uplifting_mode:
        emotion = "uplifting"
    
    if emotion in MUSIC_DATABASE:
        songs = MUSIC_DATABASE[emotion]
        
        # Filter by language if not "Any"
        if language != "Any":
            filtered_songs = [song for song in songs if song["language"] == language]
            if filtered_songs:
                return random.choice(filtered_songs)
            else:
                # If no songs in selected language, fall back to any language
                return random.choice(songs)
        else:
            return random.choice(songs)
    else:
        return random.choice(MUSIC_DATABASE["happy"])

def get_youtube_url(song, artist):
    """
    Generate YouTube search URL with song and artist
    """
    search_query = f"{song} {artist}".replace(" ", "+")
    youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
    return youtube_url

def map_deepface_emotion(deepface_emotion):
    """
    Map DeepFace's 7 emotions to our 4 categories
    DeepFace emotions: angry, disgust, fear, happy, sad, surprise, neutral
    Our emotions: happy, sad, angry, surprise
    """
    emotion_mapping = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "surprise": "surprise",
        "neutral": "happy",
        "fear": "sad",
        "disgust": "angry"
    }
    return emotion_mapping.get(deepface_emotion, "happy")

def detect_emotion_from_face(image):
    """
    Detect emotion from face using DeepFace
    Returns: detected emotion or None
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(
            img_bgr,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        # Extract dominant emotion
        if isinstance(result, list):
            result = result[0]
        
        dominant_emotion = result['dominant_emotion']
        
        # Map to our emotion categories
        mapped_emotion = map_deepface_emotion(dominant_emotion)
        
        return mapped_emotion, dominant_emotion
    
    except Exception as e:
        st.error(f"❌ Could not detect emotion from face: {e}")
        return None, None

# Main App UI
st.markdown('<h1 class="main-title">🎵 SoulSync-Emotion Based Song Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover music that matches your mood!</p>', unsafe_allow_html=True)

# Initialize session state
if 'recommended_song' not in st.session_state:
    st.session_state.recommended_song = None
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None

# Language selector in sidebar
with st.sidebar:
    st.header("🌍 Music Preferences")
    selected_language = st.selectbox(
        "Select Music Language",
        ["Any", "English", "Hindi"],
        help="Choose the language for your music recommendations"
    )
    st.markdown("---")
    st.markdown("""
    ### About the App
    This app detects your emotion and recommends music!
    
    **Features:**
    - 💬 Text analysis
    - 🎤 Voice recognition  
    - 📸 Facial emotion detection
    - 🌟 Uplifting mode
    """)

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["💬 Text Input", "🎤 Voice Input", "📸 Face Detection", "😊 Quick Emotion Select"])

# Tab 1: Text Input
with tab1:
    st.subheader("Express Your Feelings in Words")
    text_input = st.text_area(
        "How are you feeling right now?",
        placeholder="E.g., I'm feeling really happy today! or I'm so frustrated with everything...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        uplifting_mode_text = st.checkbox("🌟 Uplifting Mode", key="uplifting_text")
    
    if st.button("🎵 Find Music", type="primary", use_container_width=True):
        if text_input:
            with st.spinner("Analyzing your emotion..."):
                emotion = analyze_emotion_from_text(text_input)
                st.session_state.detected_emotion = emotion
                st.session_state.recommended_song = recommend_music(emotion, uplifting_mode_text, selected_language)
        else:
            st.warning("⚠️ Please enter some text first!")

# Tab 2: Voice Input
with tab2:
    st.subheader("Speak Your Feelings")
    st.write("Record your voice to express how you're feeling.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        uplifting_mode_voice = st.checkbox("🌟 Uplifting Mode", key="uplifting_voice")
    
    audio_input = st.audio_input("🎤 Record your voice")
    
    if audio_input is not None:
        with st.spinner("Processing your audio..."):
            audio_bytes = audio_input.read()
            recognized_text = recognize_speech_from_audio(audio_bytes)
            
            if recognized_text:
                st.success(f"📝 You said: **{recognized_text}**")
                
                with st.spinner("Analyzing your emotion..."):
                    emotion = analyze_emotion_from_text(recognized_text)
                    st.session_state.detected_emotion = emotion
                    st.session_state.recommended_song = recommend_music(emotion, uplifting_mode_voice, selected_language)

# Tab 3: Face Detection
with tab3:
    st.subheader("Detect Emotion from Your Face")
    st.write("Take a photo using your camera and we'll detect your emotion!")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        uplifting_mode_face = st.checkbox("🌟 Uplifting Mode", key="uplifting_face")
    
    camera_photo = st.camera_input("📸 Take a photo")
    
    if camera_photo is not None:
        with st.spinner("Analyzing your facial expression..."):
            # Convert uploaded image to PIL Image
            image = Image.open(camera_photo)
            
            # Detect emotion from face
            emotion, original_emotion = detect_emotion_from_face(image)
            
            if emotion:
                st.success(f"🎭 Detected facial emotion: **{original_emotion.title()}** → Mapped to: **{emotion.title()}**")
                
                st.session_state.detected_emotion = emotion
                st.session_state.recommended_song = recommend_music(emotion, uplifting_mode_face, selected_language)
            else:
                st.warning("⚠️ Could not detect a face in the image. Please try again with better lighting.")

# Tab 4: Quick Emotion Select
with tab4:
    st.subheader("Select Your Emotion")
    st.write("Choose how you're feeling right now:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("😊 Happy", use_container_width=True):
            st.session_state.detected_emotion = "happy"
            st.session_state.recommended_song = recommend_music("happy", False, selected_language)
    
    with col2:
        if st.button("😢 Sad", use_container_width=True):
            st.session_state.detected_emotion = "sad"
            st.session_state.recommended_song = recommend_music("sad", False, selected_language)
    
    with col3:
        if st.button("😠 Angry", use_container_width=True):
            st.session_state.detected_emotion = "angry"
            st.session_state.recommended_song = recommend_music("angry", False, selected_language)
    
    with col4:
        if st.button("😲 Surprised", use_container_width=True):
            st.session_state.detected_emotion = "surprise"
            st.session_state.recommended_song = recommend_music("surprise", False, selected_language)
    
    st.write("")
    uplifting_mode_quick = st.checkbox("🌟 Uplifting Mode", key="uplifting_quick")
    
    if uplifting_mode_quick and st.session_state.detected_emotion:
        st.session_state.recommended_song = recommend_music("uplifting", True, selected_language)

# Display Recommendation
st.markdown("---")

if st.session_state.recommended_song:
    st.subheader("🎵 Your Music Recommendation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        emotion_emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "surprise": "😲",
            "uplifting": "🌟"
        }
        
        detected_emotion = st.session_state.detected_emotion if st.session_state.detected_emotion else "happy"
        emoji = emotion_emoji.get(detected_emotion, "🎵")
        
        st.markdown(f"### {emoji} Detected Emotion: **{detected_emotion.title()}**")
        st.markdown(f"### 🎵 Song: **{st.session_state.recommended_song['song']}**")
        st.markdown(f"### 🎤 Artist: **{st.session_state.recommended_song['artist']}**")
        st.markdown(f"### 🌍 Language: **{st.session_state.recommended_song['language']}**")
    
    with col2:
        st.write("")
        st.write("")
        youtube_url = get_youtube_url(
            st.session_state.recommended_song['song'],
            st.session_state.recommended_song['artist']
        )
        st.link_button("▶️ Play on YouTube", youtube_url, type="primary", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> For best results with voice input, speak clearly and express your emotions!</p>
    <p>📸 <strong>Face Detection:</strong> Make sure your face is well-lit and clearly visible for accurate emotion detection.</p>
    <p>🌟 <strong>Uplifting Mode:</strong> Activate this to get positive, mood-boosting music regardless of your current emotion.</p>
</div>
""", unsafe_allow_html=True)
