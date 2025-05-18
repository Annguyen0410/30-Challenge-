import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import json
import requests # For weather API
import random # For jokes
import logging # For logging
import time # For sleep functionality
import subprocess # For more robust app opening

# --- Configuration ---
CONFIG_FILE = 'config.json'
LOG_FILE = 'assistant.log'

# --- Initialize Logging ---
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
engine = None
recognizer = None
config_data = {}

# --- Helper Functions ---
def load_config():
    """Loads configuration from the JSON file."""
    global config_data
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
        logging.info("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Please create it.")
        logging.error(f"Configuration file '{CONFIG_FILE}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Configuration file '{CONFIG_FILE}' is not valid JSON.")
        logging.error(f"Configuration file '{CONFIG_FILE}' is not valid JSON.")
        exit()

def initialize_systems():
    """Initializes speech engine and recognizer."""
    global engine, recognizer
    try:
        engine = pyttsx3.init()
        # Optional: Adjust voice properties
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[0].id) # Change index for different voices
        # engine.setProperty('rate', 180) # Speed of speech

        recognizer = sr.Recognizer()
        # Adjust recognizer sensitivity
        recognizer.energy_threshold = 4000 # Experiment with this value
        recognizer.dynamic_energy_threshold = True # Adjusts based on ambient noise
        logging.info("Speech engine and recognizer initialized.")
    except Exception as e:
        print(f"Error initializing speech systems: {e}")
        logging.error(f"Error initializing speech systems: {e}")
        exit()

def speak(text):
    """Makes the assistant speak the given text."""
    if not engine:
        print("ERROR: Speech engine not initialized.")
        logging.error("Speak function called but engine not initialized.")
        return
    print(f"Assistant: {text}")
    logging.info(f"Assistant said: {text}")
    engine.say(text)
    engine.runAndWait()

def take_command(timeout_seconds=5):
    """
    Listens for a voice command from the user and returns it as text.
    Includes timeout and fallback to offline recognition.
    """
    if not recognizer:
        print("ERROR: Recognizer not initialized.")
        logging.error("Take_command function called but recognizer not initialized.")
        return None

    with sr.Microphone() as source:
        print("Listening for command...")
        # recognizer.adjust_for_ambient_noise(source, duration=1) # Already handled by dynamic_energy_threshold
        try:
            audio = recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=7)
        except sr.WaitTimeoutError:
            # print("No command heard within the timeout period.")
            logging.info("No command heard within the timeout period.")
            return None

    try:
        print("Recognizing command...")
        # Try Google Speech Recognition (online)
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}")
        logging.info(f"User said (Google): {command}")
    except sr.UnknownValueError:
        speak("Sorry, I did not understand that.")
        logging.warning("Google Speech Recognition: UnknownValueError (could not understand audio).")
        return None
    except sr.RequestError:
        speak("Network error. Trying offline recognition.")
        logging.warning("Google Speech Recognition: RequestError (network issue).")
        # Fallback to CMU Sphinx (offline)
        try:
            command = recognizer.recognize_sphinx(audio)
            print(f"User said (Sphinx): {command}")
            logging.info(f"User said (Sphinx): {command}")
        except sr.UnknownValueError:
            speak("Offline recognition also failed to understand.")
            logging.warning("Sphinx: UnknownValueError.")
            return None
        except sr.RequestError as e_sphinx:
            speak("Offline recognition service is unavailable.")
            logging.error(f"Sphinx: RequestError: {e_sphinx}")
            return None
    except Exception as e:
        speak("An unexpected error occurred during recognition.")
        logging.error(f"Unexpected recognition error: {e}")
        return None

    return command.lower()

def greet_user():
    """Greets the user based on the time of day and configured name."""
    user_name = config_data.get("user_name", "User")
    assistant_name = config_data.get("assistant_name", "Assistant")
    hour = datetime.datetime.now().hour
    greeting = ""
    if 0 <= hour < 12:
        greeting = f"Good morning {user_name}!"
    elif 12 <= hour < 18:
        greeting = f"Good afternoon {user_name}!"
    else:
        greeting = f"Good evening {user_name}!"
    speak(f"{greeting} I am {assistant_name}. How can I assist you today after you say my wake word?")

def get_weather():
    """Fetches and speaks the current weather for the default city."""
    api_key = config_data.get("weather_api_key")
    city = config_data.get("default_city", "London") # Default to London if not in config

    if not api_key:
        speak("Weather API key not configured. I can't fetch the weather.")
        logging.warning("Weather API key missing in config.")
        return

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}&units=metric" # Using metric units

    try:
        response = requests.get(complete_url)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        weather_data = response.json()

        if weather_data["cod"] != "404":
            main_data = weather_data["main"]
            current_temperature = main_data["temp"]
            current_pressure = main_data["pressure"]
            current_humidity = main_data["humidity"]
            weather_description = weather_data["weather"][0]["description"]
            speak(f"The weather in {city} is currently {weather_description} "
                  f"with a temperature of {current_temperature:.1f} degrees Celsius, "
                  f"humidity at {current_humidity} percent, "
                  f"and atmospheric pressure of {current_pressure} hectopascals.")
        else:
            speak(f"Sorry, I couldn't find weather data for {city}.")
            logging.warning(f"Weather data not found for {city} (API response code 404).")
    except requests.exceptions.RequestException as e:
        speak("Sorry, I couldn't connect to the weather service.")
        logging.error(f"Weather API request error: {e}")
    except KeyError:
        speak("Received incomplete weather data. Please check the API response or city name.")
        logging.error("KeyError while parsing weather data.")


def tell_joke():
    """Tells a random joke from the configured list."""
    jokes = config_data.get("jokes", [])
    if jokes:
        joke = random.choice(jokes)
        speak(joke)
    else:
        speak("I don't have any jokes configured right now.")
        logging.info("Attempted to tell a joke, but no jokes found in config.")

def open_application(command_text):
    """Opens an application based on the command and configuration."""
    apps = config_data.get("applications", {})
    opened = False
    for app_name, app_path in apps.items():
        if app_name in command_text:
            try:
                speak(f"Opening {app_name}")
                # Using subprocess.Popen for non-blocking opening (more robust)
                # For simple .exe, os.system might be fine, but Popen is generally better.
                if os.path.exists(app_path):
                    subprocess.Popen([app_path]) # For .exe files or scripts
                elif os.path.exists(app_name): # If app_path is just the command name (e.g. 'notepad')
                     subprocess.Popen([app_name], shell=True) # shell=True for system commands
                else: # Try os.startfile for registered file types or URLs
                    os.startfile(app_path) # More Windows-specific, good for documents/URLs
                opened = True
                break
            except FileNotFoundError:
                speak(f"Sorry, I could not find {app_name} at the path: {app_path}")
                logging.error(f"Application path not found for {app_name}: {app_path}")
            except Exception as e:
                speak(f"Sorry, I encountered an error trying to open {app_name}.")
                logging.error(f"Error opening {app_name} at {app_path}: {e}")
            break # Only try to open the first match
    if not opened and "open" in command_text: # If 'open' was in command but no app matched
        speak("I'm not sure which application you want to open from my list.")


# --- Main Assistant Logic ---
def respond(command):
    """Responds to different commands."""
    assistant_name = config_data.get("assistant_name", "Assistant")

    if any(greeting in command for greeting in ['hello', 'hi', 'hey']):
        speak(f"Hello {config_data.get('user_name', 'User')}! How can I help?")

    elif 'your name' in command:
        speak(f"My name is {assistant_name}.")

    elif 'time' in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif 'date' in command:
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y") # E.g., Monday, July 26, 2023
        speak(f"Today is {current_date}")

    elif 'search for' in command or 'google' in command:
        search_query = command.replace('search for', '').replace('google', '').strip()
        if not search_query:
            speak("What would you like to search for?")
            search_query = take_command()
        if search_query and search_query != "none": # take_command returns "none" as string on failure with Sphinx
            speak(f"Searching Google for {search_query}")
            webbrowser.open(f"https://www.google.com/search?q={search_query.replace(' ', '+')}")
        else:
            speak("I didn't catch what to search for.")

    elif 'open' in command:
        open_application(command)

    elif 'weather' in command:
        get_weather()

    elif 'joke' in command or 'tell me a joke' in command:
        tell_joke()
    
    elif 'sleep' in command or 'go to sleep' in command:
        speak("Understood. I'll go to sleep for a bit. Say my wake word to reactivate me.")
        return "sleep" # Signal to the main loop to enter sleep mode

    elif any(exit_cmd in command for exit_cmd in ['bye', 'exit', 'quit', 'goodbye', 'shutdown']):
        speak("Goodbye! Have a great day.")
        logging.info("Assistant shutting down by user command.")
        return "exit" # Signal to the main loop to exit

    else:
        speak("I'm sorry, I don't understand that command. Can you please repeat or try something else?")
        logging.info(f"Unknown command: {command}")
    return None # Continue listening

def listen_for_wake_word(wake_word, timeout_seconds=5):
    """Listens specifically for the wake word."""
    if not recognizer: return False
    with sr.Microphone() as source:
        print(f"Listening for wake word ('{wake_word}')...")
        # recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            return False # No audio detected

        try:
            # Using Sphinx for wake word detection as it's offline and faster for short phrases
            # and doesn't require constant Google API calls.
            # You can also try recognizer.recognize_google(audio) if Sphinx is problematic
            text = recognizer.recognize_sphinx(audio)
            # text = recognizer.recognize_google(audio) # Alternative
            print(f"Heard: {text}") # For debugging wake word recognition
            logging.debug(f"Wake word recognizer heard: {text}")
            return wake_word in text.lower()
        except sr.UnknownValueError:
            # print("Sphinx: Could not understand audio for wake word")
            logging.debug("Sphinx: Could not understand audio for wake word")
            return False
        except sr.RequestError as e:
            print(f"Sphinx error for wake word; {e}")
            logging.error(f"Sphinx error for wake word; {e}")
            return False # If Sphinx has an issue, we can't detect wake word

def run_assistant():
    """Main function to run the assistant."""
    load_config()
    initialize_systems()

    wake_word = config_data.get("wake_word", "assistant").lower()
    greet_user()
    
    active_listening = False # Start in standby mode

    while True:
        if not active_listening:
            if listen_for_wake_word(wake_word):
                speak(f"Yes {config_data.get('user_name', 'Boss')}?")
                active_listening = True
                last_interaction_time = time.time()
            else:
                # Optional: Add a small delay to prevent constant microphone use if wake word detection is CPU intensive
                # time.sleep(0.1) 
                continue # Continue listening for wake word
        
        if active_listening:
            # Timeout for active listening if no command is given
            if time.time() - last_interaction_time > 30: # 30 seconds timeout
                speak("No commands for a while. Going back to sleep. Say my wake word to activate me.")
                active_listening = False
                continue

            command = take_command()
            if command:
                last_interaction_time = time.time() # Reset timer on successful command
                action = respond(command)
                if action == "exit":
                    break
                elif action == "sleep":
                    active_listening = False
            elif command is None and active_listening: # take_command timed out or failed
                # if no command for a while, it might go to sleep (handled by timeout above)
                # or you can ask if user still there
                # speak("Are you still there?")
                pass


if __name__ == "__main__":
    run_assistant()