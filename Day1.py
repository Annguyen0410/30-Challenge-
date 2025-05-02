import math
import sys
import os
import json # For saving/loading history and state

# --- Utility Functions ---

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Mathematical Functions ---
# (These remain largely independent of the calculator state)

def add(x, y): return x + y
def subtract(x, y): return x - y
def multiply(x, y): return x * y
def divide(x, y):
    if y == 0: raise ValueError("Error: Division by zero is not allowed.")
    return x / y
def power(x, y): return x ** y
def modulo(x, y):
    if y == 0: raise ValueError("Error: Modulo by zero is not allowed.")
    return x % y

def square_root(x):
    if x < 0: raise ValueError("Error: Cannot calculate the square root of a negative number.")
    return math.sqrt(x)
def cube_root(x):
    # Handles negative numbers correctly for cube root
    return x**(1/3) if x >= 0 else -(-x)**(1/3)

def factorial(x):
    if x < 0: raise ValueError("Error: Factorial is not defined for negative numbers.")
    if not isinstance(x, int) and not x.is_integer():
         raise ValueError("Error: Factorial is only defined for non-negative integers.")
    if x > 170: # Prevent OverflowError for large factorials
        raise OverflowError("Error: Factorial input too large, causes overflow.")
    return math.factorial(int(x))

# --- Trig Functions (Degrees) ---
def sin_degrees(x): return math.sin(math.radians(x))
def cos_degrees(x): return math.cos(math.radians(x))
def tan_degrees(x):
    if (x % 180) == 90: raise ValueError(f"Error: Tangent is undefined for {x} degrees.")
    return math.tan(math.radians(x))

def asin_degrees(x):
    if not -1 <= x <= 1: raise ValueError("Error: arcsin input must be between -1 and 1.")
    return math.degrees(math.asin(x))
def acos_degrees(x):
    if not -1 <= x <= 1: raise ValueError("Error: arccos input must be between -1 and 1.")
    return math.degrees(math.acos(x))
def atan_degrees(x): return math.degrees(math.atan(x))

# --- Logarithmic Functions ---
def natural_log(x):
    if x <= 0: raise ValueError("Error: Natural logarithm is only defined for positive numbers.")
    return math.log(x)
def base10_log(x):
    if x <= 0: raise ValueError("Error: Base-10 logarithm is only defined for positive numbers.")
    return math.log10(x)

# --- Calculator Class ---

class Calculator:
    def __init__(self, history_file="calc_history.json"):
        self.memory = 0.0
        self.last_result = None
        self.history = []
        self.history_file = history_file
        self._load_state() # Load previous state if available

        self.operations = {
            # Basic Arithmetic
            '1': (add, '+', 2), 'add': (add, '+', 2),
            '2': (subtract, '-', 2), 'sub': (subtract, '-', 2),
            '3': (multiply, '*', 2), 'mul': (multiply, '*', 2),
            '4': (divide, '/', 2), 'div': (divide, '/', 2),
            '5': (modulo, '%', 2), 'mod': (modulo, '%', 2),
            # Exponents & Roots
            '6': (power, '^', 2), 'pow': (power, '^', 2),
            '7': (square_root, 'sqrt', 1), 'sqrt': (square_root, 'sqrt', 1),
            '8': (cube_root, 'cbrt', 1), 'cbrt': (cube_root, 'cbrt', 1),
            # Factorial
            '9': (factorial, '!', 1), 'fact': (factorial, '!', 1),
            # Trigonometry (Degrees)
            '10': (sin_degrees, 'sin', 1), 'sin': (sin_degrees, 'sin', 1),
            '11': (cos_degrees, 'cos', 1), 'cos': (cos_degrees, 'cos', 1),
            '12': (tan_degrees, 'tan', 1), 'tan': (tan_degrees, 'tan', 1),
            '13': (asin_degrees, 'asin', 1), 'asin': (asin_degrees, 'asin', 1),
            '14': (acos_degrees, 'acos', 1), 'acos': (acos_degrees, 'acos', 1),
            '15': (atan_degrees, 'atan', 1), 'atan': (atan_degrees, 'atan', 1),
            # Logarithms
            '16': (natural_log, 'ln', 1), 'ln': (natural_log, 'ln', 1),
            '17': (base10_log, 'log10', 1), 'log10': (base10_log, 'log10', 1),
        }

    def _save_state(self):
        try:
            state = {
                'memory': self.memory,
                'last_result': self.last_result,
                'history': self.history[-100:] # Save last 100 history items
            }
            with open(self.history_file, 'w') as f:
                json.dump(state, f)
        except IOError as e:
            print(f"\nWarning: Could not save state to {self.history_file}: {e}")
        except Exception as e:
            print(f"\nWarning: An unexpected error occurred while saving state: {e}")


    def _load_state(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    state = json.load(f)
                    self.memory = state.get('memory', 0.0)
                    self.last_result = state.get('last_result', None)
                    self.history = state.get('history', [])
                    print(f"Loaded previous state from {self.history_file}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"\nWarning: Could not load state from {self.history_file}: {e}")
            # Reset to defaults if file is corrupted or unreadable
            self.memory = 0.0
            self.last_result = None
            self.history = []
        except Exception as e:
            print(f"\nWarning: An unexpected error occurred while loading state: {e}")
            self.memory = 0.0
            self.last_result = None
            self.history = []


    def get_number_input(self, prompt):
        while True:
            try:
                user_input = input(prompt).strip().lower()

                if user_input == 'quit':
                    self._save_state()
                    print("\nCalculator state saved. Exiting. Goodbye!")
                    sys.exit()

                if user_input == 'ans':
                    if self.last_result is not None:
                        print(f"Using ans: {self.last_result:.4f}")
                        return self.last_result
                    else:
                        print("Error: No previous result ('ans') available.")
                        continue

                if user_input == 'mr':
                    print(f"Using memory: {self.memory:.4f}")
                    return self.memory

                if user_input == 'pi': return math.pi
                if user_input == 'e': return math.e

                return float(user_input)

            except ValueError:
                print("Invalid input. Enter a number, 'ans', 'mr', 'pi', 'e', or 'quit'.")


    def display_history(self):
        if not self.history:
            print("No history yet.")
            return
        print("\n--- Calculation History ---")
        # Display most recent first
        for i, item in enumerate(reversed(self.history)):
            print(f"{len(self.history)-i}: {item}")
        print("-------------------------")

    def run(self):
        clear_screen()
        print("Advanced Calculator")
        print("Enter command names (e.g., 'add', 'sqrt') or numbers.")
        print("Type 'help' for command list, 'quit' to exit.")

        while True:
            print(f"\n[Mem: {self.memory:.4f}] [Ans: {'None' if self.last_result is None else f'{self.last_result:.4f}'}]")
            choice = input("Enter command or operation number: ").strip().lower()

            # --- Handle Non-Calculation Commands ---
            if choice == 'quit':
                self._save_state()
                print("\nCalculator state saved. Exiting. Goodbye!")
                break
            elif choice in ('hist', 'history'):
                self.display_history()
                continue
            elif choice == 'clear':
                clear_screen()
                print("Advanced Calculator")
                print("Type 'help' for command list, 'quit' to exit.")
                continue
            elif choice == 'help':
                self.display_help()
                continue
            elif choice == 'mc':
                self.memory = 0.0
                print("Memory cleared.")
                continue
            elif choice == 'mr': # Display memory value if typed directly
                 print(f"Memory Contains: {self.memory:.4f}")
                 continue
            elif choice in ('ms', 'm+', 'm-'):
                if self.last_result is None:
                    print("Error: No last result available for memory operation ('ms', 'm+', 'm-').")
                else:
                    if choice == 'ms':
                        self.memory = self.last_result
                        print(f"{self.last_result:.4f} stored in memory.")
                    elif choice == 'm+':
                        self.memory += self.last_result
                        print(f"{self.last_result:.4f} added to memory. New value: {self.memory:.4f}")
                    elif choice == 'm-':
                        self.memory -= self.last_result
                        print(f"{self.last_result:.4f} subtracted from memory. New value: {self.memory:.4f}")
                continue

            # --- Handle Calculation Operations ---
            elif choice in self.operations:
                operation_func, symbol, num_args = self.operations[choice]

                try:
                    num_prompt = "Enter number (or 'ans', 'mr', 'pi', 'e'): "
                    num1 = self.get_number_input(num_prompt if num_args == 1 else "Enter first number: ")

                    result = 0
                    result_str = ""

                    if num_args == 1:
                        result = operation_func(num1)
                        op_name = symbol if symbol != '!' else '' # Use symbol like sqrt, sin, !, etc.
                        num1_display = int(num1) if symbol == '!' else f"{num1:.4f}"
                        result_display = f"{result:.4f}" if isinstance(result, float) else str(result)

                        result_str = f"{op_name}({num1_display}){'' if symbol != '!' else symbol} = {result_display}"


                    else: # num_args == 2
                        num2 = self.get_number_input("Enter second number: ")
                        result = operation_func(num1, num2)
                        result_str = f"{num1:.4f} {symbol} {num2:.4f} = {result:.4f}"

                    print(f"Result: {result_str}")
                    self.history.append(result_str)
                    self.last_result = result

                except (ValueError, TypeError, OverflowError) as e:
                    print(e)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

            else:
                print(f"Unknown command: '{choice}'. Type 'help' for available commands.")

    def display_help(self):
         print("\n--- Available Commands ---")
         print("Core Operations:")
         print("  1/add   2/sub   3/mul    4/div    5/mod")
         print("Exponents/Roots:")
         print("  6/pow   7/sqrt  8/cbrt")
         print("Factorial:")
         print("  9/fact (!)")
         print("Trigonometry (Degrees):")
         print(" 10/sin  11/cos  12/tan   13/asin  14/acos  15/atan")
         print("Logarithms:")
         print(" 16/ln   17/log10")
         print("Memory:")
         print("  ms: Store Ans   mr: Recall Mem   mc: Clear Mem")
         print("  m+: Add Ans     m-: Sub Ans")
         print("Other:")
         print("  ans: Use Last Result in input")
         print("  pi / e: Use constants in input")
         print("  hist: Show History    clear: Clear Screen")
         print("  help: Show this help  quit: Exit Calculator")
         print("------------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    calc = Calculator()
    try:
        calc.run()
    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        calc._save_state()
        print("\nCalculator state saved. Exiting due to interrupt.")
    except Exception as main_err: # Catch unexpected errors in run loop
        print(f"\nA critical error occurred: {main_err}")
        calc._save_state()
        print("Attempted to save state before exiting.")