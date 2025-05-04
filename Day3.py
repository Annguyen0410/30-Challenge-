import re
import random
import operator
import datetime

class EnhancedChatBot:
    def __init__(self):
        self.user_name = None
        self.memory = {} # Simple key-value memory
        self.conversation_history = [] # Track recent interactions

        # --- Predefined Data ---
        self.knowledge_base = {
            "what is the capital of france": "The capital of France is Paris.",
            "what is the tallest mountain": "Mount Everest is the tallest mountain in the world.",
            "who wrote hamlet": "Hamlet was written by William Shakespeare.",
            "what is h2o": "H2O is the chemical formula for water.",
            "python programming language": "Python is a popular high-level, interpreted programming language known for its readability.",
        }

        # --- Pattern Mapping ---
        # Order can matter, more specific patterns should come first
        self.response_patterns = [
            # --- Core Interaction & Personalization ---
            (r"\b(hello|hi|hey|greetings|yo|sup)\b", self.handle_greeting),
            (r"\bhow are you\b.*", self.handle_status_query),
            (r".*(how about you|and you)\b.*", self.handle_reciprocal_status),
            (r"\bmy name is (.*)", self.handle_set_name),
            (r"\bcall me (.*)", self.handle_set_name), # Alternative phrasing
            (r"\b(what is my name|do you know my name|who am i)\b", self.handle_get_name),
            (r"\b(what is your name|who are you)\b.*", self.handle_bot_identity),
            (r"\b(help|assist|support|what can you do|capabilities)\b.*", self.handle_help_request),
            (r"\b(thanks|thank you|ty|thx)\b.*", self.handle_thanks),
            (r"\b(bye|quit|exit|goodbye|see ya|later)\b", self.handle_exit), # Must return specific marker

            # --- Information & Q&A ---
            (r".*weather.*", self.handle_weather),
            (r".*(time|what time is it).*", self.handle_time),
            (r".*(date|what is today).*", self.handle_date),
            (r".*(joke|tell me something funny).*", self.handle_joke),
            (r".*favorite (color|food|movie|book|music).*", self.handle_favorites),
            (r"\b(who created you|who made you|developer)\b", self.handle_creator),
            # Simple knowledge base lookup (attempt exact match first)
            (r".*", self.handle_knowledge_lookup), # Will be checked specifically

            # --- Memory ---
            (r"\bremember that (.*) is (.*)", self.handle_remember),
            (r"\bremember (.*)", self.handle_remember_simple), # Remember single fact
            (r"\b(what is|what do you know about|recall) (.*)", self.handle_recall),
            (r"\bforget (.*)", self.handle_forget),
            (r"\b(what have we talked about|history)\b", self.handle_history),

            # --- Calculation ---
            (r".*(calculate|compute|solve|what is) ([-+]?\d+(\.\d+)?)\s*([\+\-\*\/])\s*([-+]?\d+(\.\d+)?).*", self.handle_calculation),

            # --- User State & Interaction ---
            (r"\b(i feel|i'm feeling) (.*)", self.handle_user_feeling),
            (r"\b(i like|i love|i enjoy) (.*)", self.handle_user_likes),
            (r"\b(repeat|say) (.*)", self.handle_repeat),
            (r"\b(flip a coin|coin flip)\b", self.handle_coin_flip),
            (r"\b(roll a die|roll dice)\b", self.handle_dice_roll),
            (r"\b(clear memory|forget everything)\b", self.handle_clear_memory),

            # --- Default / Fallback ---
            # The main loop handles default if no pattern matches
        ]

        # --- Response Sets ---
        self.default_responses = [
            "I'm not sure I quite understand. Could you perhaps rephrase that?",
            "That's an interesting point! Could you elaborate a bit?",
            "My apologies, I didn't catch that. Can you try asking in a different way?",
            "I'm still under development and learning. That's a bit beyond me right now.",
            "Hmm, let me process that... Unfortunately, I don't have a specific response.",
            "Could you tell me more about that?",
            "Is there another way I can help with that topic?",
        ]
        self.teach_me_prompts = [
            "I don't know about that. Could you teach me? For example, 'remember that [topic] is [information]'.",
            "That's new to me! If you want me to remember, tell me like: 'remember that [key] is [value]'.",
            "I haven't learned about that yet. Would you like to add it to my memory?",
        ]
        self.exit_phrases = [
            "Goodbye! Have a great day!", "Farewell! Hope to chat again soon.",
            "Bye bye!", "Catch you later!", "See you next time!",
        ]
        self.ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

    # --- Core Logic ---
    def get_response(self, user_input):
        user_input_lower = user_input.lower().strip()
        self.conversation_history.append(("user", user_input))

        # 1. Check for exact match in knowledge base
        if user_input_lower in self.knowledge_base:
             response = self.knowledge_base[user_input_lower]
             self.conversation_history.append(("bot", response))
             return response

        # 2. Iterate through regex patterns
        for pattern, handler_func in self.response_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                # Special check for the knowledge handler to avoid matching everything too early
                if handler_func == self.handle_knowledge_lookup and not self._is_potential_knowledge_query(user_input_lower):
                    continue # Skip if it doesn't look like a knowledge query

                response = handler_func(match, user_input)
                # Handle exit signal
                if response == "##EXIT##":
                    return self.handle_exit(match, user_input) # Get actual exit phrase

                if response: # Ensure handler returned a valid response
                    self.conversation_history.append(("bot", response))
                    return response

        # 3. If no pattern matched, use default response
        response = random.choice(self.default_responses)
        # Occasionally ask to be taught
        if random.random() < 0.15: # 15% chance
            response += " " + random.choice(self.teach_me_prompts)

        self.conversation_history.append(("bot", response))
        return response

    def _is_potential_knowledge_query(self, text):
        # Simple heuristic: does it start with question words or contain 'about'?
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'tell me about']
        return any(text.startswith(qw) for qw in question_words) or ' about ' in text


    # --- Handler Functions ---
    # (Handlers should return the response string, or "##EXIT##" for exit command)

    def handle_greeting(self, match, user_input):
        greetings = ["Hi", "Hello", "Hey", "Greetings"]
        response = f"{random.choice(greetings)}"
        if self.user_name:
            response += f" {self.user_name}!"
        else:
            response += " there!"
        response += f" {random.choice(['How can I assist you today?', 'How can I help?', 'What can I do for you?'])}"
        return response

    def handle_status_query(self, match, user_input):
        return random.choice([
            "I'm functioning optimally, thank you! Ready for your requests.",
            "I'm doing great! Just processing bits and bytes. How about yourself?",
            "As an AI, I don't have feelings, but I'm operating perfectly!",
            "Running smoothly and ready to help!",
        ])

    def handle_reciprocal_status(self, match, user_input):
         return random.choice([
            "I'm doing well, thanks for asking back!",
            "Functioning perfectly over here!",
            "Just processing information as usual. Thanks!",
         ])

    def handle_set_name(self, match, user_input):
        name = match.group(1).strip().title()
        if name and len(name) < 30: # Basic validation
            self.user_name = name
            return random.choice([
                f"Nice to meet you, {self.user_name}!",
                f"Got it, I'll remember your name is {self.user_name}.",
                f"Pleasure to make your acquaintance, {self.user_name}!",
                f"{self.user_name}, noted!",
            ])
        else:
            return "That doesn't seem like a valid name. Could you try again?"

    def handle_get_name(self, match, user_input):
        if self.user_name:
            return random.choice([
                f"Your name is {self.user_name}, if I remember correctly.",
                f"I believe you told me your name is {self.user_name}.",
                f"Yes, you're {self.user_name}.",
                f"I have your name down as {self.user_name}.",
            ])
        else:
            return random.choice([
                "I don't think you've told me your name yet.",
                "I don't seem to have your name stored. What is it?",
                "You haven't mentioned your name. Feel free to tell me!",
            ])

    def handle_bot_identity(self, match, user_input):
        return random.choice([
            "I'm a chatbot designed to assist with information and simple tasks.",
            "You can call me ChatBot. I'm an AI assistant.",
            "I'm a virtual assistant coded in Python.",
            "I am a language model here to chat!",
        ])

    def handle_help_request(self, match, user_input):
        features = [
            "chatting", "remembering facts ('remember that X is Y')", "recalling facts ('what is X')",
            "telling jokes", "providing the date/time", "basic calculations ('calculate 5 * 3')",
            "flipping a coin", "rolling a die",
            "I can also try to answer general knowledge questions."
        ]
        random.shuffle(features)
        return f"I can help with: {', '.join(features[:4])}. What would you like to do?"


    def handle_thanks(self, match, user_input):
        return random.choice([
            "You're very welcome!", "No problem at all!", "Anytime!",
            "Glad I could assist!", "Happy to help!", "My pleasure!",
        ])

    def handle_exit(self, match, user_input):
        # This function is now primarily called by get_response after detecting "##EXIT##"
        # Or directly if the exit pattern is the *only* thing matched.
        response = random.choice(self.exit_phrases)
        if self.user_name:
            # Add name without doubling punctuation if already present
             if response.endswith('.'): response = response[:-1] + f", {self.user_name}."
             elif response.endswith('!'): response = response[:-1] + f", {self.user_name}!"
             else: response = response + f", {self.user_name}"

        return response # Return the final exit phrase

    def handle_weather(self, match, user_input):
        return random.choice([
            "I can't access real-time weather data. You might want to check a weather website or app!",
            "Predicting the weather is beyond my capabilities right now, sorry!",
            "I'm not equipped to check the weather forecast.",
        ])

    def handle_time(self, match, user_input):
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    def handle_date(self, match, user_input):
        now = datetime.datetime.now()
        return f"Today's date is {now.strftime('%A, %B %d, %Y')}."

    def handle_joke(self, match, user_input):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call fake spaghetti? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a lazy kangaroo? Pouch potato!",
            "Why did the bicycle fall over? Because it was two tired!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fish wearing a bowtie? Sofishticated!",
            "Why was the math book sad? Because it had too many problems.",
        ]
        return random.choice(jokes)

    def handle_favorites(self, match, user_input):
        topic = match.group(1)
        responses = {
            "color": "I find shades of digital blue quite appealing.",
            "food": "As an AI, I feast on data!",
            "movie": "I process information, so maybe something like '2001: A Space Odyssey'?",
            "book": "I enjoy parsing through vast amounts of text, like encyclopedias!",
            "music": "I like the sound of algorithms sorting efficiently.",
        }
        return responses.get(topic, f"I don't have personal preferences like a favorite {topic}, but I appreciate the concept!")

    def handle_creator(self, match, user_input):
        return random.choice([
            "I was created by a programmer experimenting with Python and AI concepts.",
            "My origins are in code, crafted by a developer.",
            "A human developer wrote the code that brings me to life.",
        ])

    def handle_knowledge_lookup(self, match, user_input):
        # This handler is called if the input *looks* like a question
        # but didn't match an exact knowledge base entry or other specific pattern.
        # We can try a simple substring search in our knowledge base keys.
        user_input_lower = user_input.lower().strip()
        for key, value in self.knowledge_base.items():
            # Check if significant parts of the input match a key
            # (This is very basic keyword spotting)
            keywords = [word for word in user_input_lower.split() if len(word) > 3] # Ignore small words
            key_words = [word for word in key.split() if len(word) > 3]
            # Require a certain number of keyword overlaps
            if len(keywords) > 1 and len(key_words) > 1:
                 common_words = set(keywords) & set(key_words)
                 # Adjust threshold based on lengths? For now, simple check:
                 if len(common_words) >= min(len(keywords), len(key_words)) * 0.6: # Need 60% overlap
                      return value # Return the answer if a likely match is found

        # If no near match found in knowledge base
        return None # Signal that this handler didn't find a specific answer


    def handle_remember(self, match, user_input):
        key = match.group(1).strip().lower() # Store keys in lowercase for easier recall
        value = match.group(2).strip()
        if key and value:
            self.memory[key] = value
            return random.choice([
                f"Okay, I'll remember that {key} is {value}.",
                f"Got it: '{key}' means '{value}'. Stored.",
                f"Information saved: {key} -> {value}.",
            ])
        else:
            return "Please provide both the item and what I should remember about it (e.g., 'remember that my cat is fluffy')."

    def handle_remember_simple(self, match, user_input):
        fact = match.group(1).strip()
        # Try to find a logical subject/object if possible (very basic)
        parts = fact.split(' is ', 1)
        if len(parts) == 2:
             key, value = parts[0].strip().lower(), parts[1].strip()
        else:
             # Default: use the whole phrase as key, value is 'True' or similar
             key, value = fact.lower(), "something you told me to remember"

        if key:
             self.memory[key] = value
             return f"Okay, I'll try to remember: '{fact}'."
        else:
             return "What should I remember?"


    def handle_recall(self, match, user_input):
        key = match.group(2).strip().lower() # Match recall key in lowercase
        if key in self.memory:
            value = self.memory[key]
            return random.choice([
                f"Ah yes, I remember that {key} is {value}.",
                f"Based on what you told me, {key} is {value}.",
                f"Recall successful: {key} -> {value}.",
            ])
        # Check knowledge base as a fallback
        elif key in self.knowledge_base:
            return self.knowledge_base[key]
        else:
            # Try partial match in memory keys
            possible_matches = [k for k in self.memory if key in k]
            if possible_matches:
                return f"I don't know about '{key}' specifically, but I remember things related to: {', '.join(possible_matches[:3])}."

            return random.choice([
                f"I don't seem to have anything stored specifically about '{key}'.",
                f"Hmm, my memory banks draw a blank on '{key}'.",
                f"I haven't learned about '{key}' yet.",
            ])

    def handle_forget(self, match, user_input):
        key = match.group(1).strip().lower()
        if key in self.memory:
            value = self.memory.pop(key) # Use pop to remove and get value
            return random.choice([
                f"Okay, I've forgotten that {key} was {value}.",
                f"Information about '{key}' has been removed from my memory.",
                f"Consider it forgotten: {key}.",
            ])
        else:
            return f"I don't have anything stored about '{key}' to forget."

    def handle_history(self, match, user_input):
         if not self.conversation_history:
              return "We haven't started talking yet!"
         # Show last few interactions
         history_str = "Here's a bit of our recent chat:\n"
         # Limit history length display
         start_index = max(0, len(self.conversation_history) - 6) # Show last 6 entries max (3 user, 3 bot)
         for sender, message in self.conversation_history[start_index:-1]: # Exclude current request
             prefix = "You" if sender == "user" else "Bot"
             history_str += f"- {prefix}: {message}\n"
         return history_str.strip()

    def handle_calculation(self, match, user_input):
        try:
            # Groups: 1: Prefix, 2: Num1, 4: Operator, 5: Num2
            num1_str = match.group(2)
            op_symbol = match.group(4)
            num2_str = match.group(5)

            num1 = float(num1_str)
            num2 = float(num2_str)

            if op_symbol in self.ops:
                if op_symbol == '/' and num2 == 0:
                    return "Oops! Division by zero is not allowed."
                operation = self.ops[op_symbol]
                result = operation(num1, num2)
                # Format result nicely
                if result == int(result): result = int(result)
                else: result = round(result, 4) # Round floats

                # Use original numbers in string for clarity
                return f"{num1_str} {op_symbol} {num2_str} equals {result}."
            else:
                # This case should theoretically not be reached due to regex
                return "Sorry, I only support basic arithmetic operations: +, -, *, /."
        except ValueError:
            return "Hmm, I had trouble understanding those numbers."
        except Exception as e:
            print(f"Calculation Error: {e}") # Log for debugging
            return "Sorry, an error occurred during the calculation."

    def handle_user_feeling(self, match, user_input):
        feeling = match.group(2).strip().lower()
        positive_words = ["good", "great", "happy", "awesome", "fantastic", "well", "excited", "fine", "ok", "better"]
        negative_words = ["sad", "bad", "terrible", "awful", "down", "not good", "upset", "angry", "stressed", "tired", "sick"]
        response = ""

        if any(word in feeling for word in positive_words):
            response = random.choice([
                "That's wonderful to hear!", "Great!", f"I'm glad you're feeling {feeling}!",
                "Awesome news!", "Good to know!",
            ])
        elif any(word in feeling for word in negative_words):
            response = random.choice([
                f"Oh no, I'm sorry to hear you're feeling {feeling}.",
                "I hope things improve for you soon.",
                "That sounds tough. Sending positive thoughts.",
                "Take care of yourself.",
                f"It's okay to feel {feeling} sometimes.",
            ])
        else:
            response = random.choice([
                f"Thanks for sharing that you feel {feeling}.",
                f"Understood. Feelings can be complex.",
                f"Okay, noted you're feeling {feeling}.",
            ])

        if self.user_name:
             response += f" {self.user_name}."

        return response


    def handle_user_likes(self, match, user_input):
        liked_item = match.group(2).strip()
        return random.choice([
            f"{liked_item.capitalize()} sounds interesting!",
            f"It's nice that you enjoy {liked_item}.",
            f"Ah, {liked_item}. Good to know!",
            f"Thanks for sharing your interest in {liked_item}.",
        ])

    def handle_repeat(self, match, user_input):
        phrase = match.group(2).strip()
        if phrase:
            return f"Okay, you said: {phrase}"
        else:
            return "What would you like me to repeat?"

    def handle_coin_flip(self, match, user_input):
        result = random.choice(["Heads", "Tails"])
        return f"I flipped a coin, and it landed on: **{result}**!"

    def handle_dice_roll(self, match, user_input):
        result = random.randint(1, 6)
        return f"I rolled a six-sided die, and it shows: **{result}**!"

    def handle_clear_memory(self, match, user_input):
        num_items = len(self.memory)
        self.memory.clear()
        return f"Okay, I've cleared my short-term memory. I forgot {num_items} item(s)."


    # --- Control Flow ---
    def is_exit_command(self, user_input):
        # This check is now less critical as handle_exit signals exit status
        # but can be kept for the main loop's initial check.
        exit_pattern = r"^\s*(bye|quit|exit|goodbye|see ya|later)\s*$" # Match if it's the main command
        return re.search(exit_pattern, user_input.lower()) is not None

    def run(self):
        greeting = self.handle_greeting(None, "") # Get initial greeting
        print(f"Chatbot: {greeting}")
        print("Chatbot: (Type 'bye', 'quit', or 'exit' to end the chat)")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                # Check for exit command *first* before general processing
                if self.is_exit_command(user_input):
                    exit_response = self.handle_exit(None, user_input)
                    print(f"Chatbot: {exit_response}")
                    break

                response = self.get_response(user_input) # This now handles internal exit signalling too

                # The get_response method itself now calls handle_exit if needed and returns the phrase
                print(f"Chatbot: {response}")

            except (EOFError, KeyboardInterrupt):
                farewell = random.choice(self.exit_phrases)
                if self.user_name:
                   if farewell.endswith('.'): farewell = farewell[:-1] + f", {self.user_name}."
                   elif farewell.endswith('!'): farewell = farewell[:-1] + f", {self.user_name}!"
                   else: farewell = farewell + f", {self.user_name}"
                print(f"\nChatbot: {farewell}")
                break
            except Exception as e:
                 # Log the error for debugging
                 print(f"\n[ERROR] An unexpected error occurred: {e}")
                 import traceback
                 traceback.print_exc() # Print detailed traceback
                 print("Chatbot: Oops! Something went wrong on my end. Let's try to continue...")


if __name__ == "__main__":
    bot = EnhancedChatBot()
    bot.run()