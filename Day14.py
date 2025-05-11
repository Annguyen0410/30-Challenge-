import math
import random
import time
import tkinter as tk
from tkinter import messagebox, ttk
import copy # For deepcopy

WIN_SCORE = 10

THEMES = {
    "Classic": {
        "bg": "SystemButtonFace",
        "fg": "black",
        "board_frame_bg": "SystemButtonFace",
        "button_bg": "SystemButtonFace", # Default button color
        "button_active_bg": "grey90",
        "X_color": "blue",
        "O_color": "red",
        "win_highlight_bg": "lightgreen",
        "hint_highlight_bg": "yellow",
        "text_main": "black",
        "text_button": "black",
        "accent_button_bg": "dodgerblue",
        "accent_button_fg": "white"
    },
    "Dark": {
        "bg": "grey20",
        "fg": "white",
        "board_frame_bg": "grey25",
        "button_bg": "grey30",
        "button_active_bg": "grey40",
        "X_color": "cyan",
        "O_color": "orange",
        "win_highlight_bg": "darkgreen",
        "hint_highlight_bg": "gold",
        "text_main": "white",
        "text_button": "white",
        "accent_button_bg": "deepskyblue",
        "accent_button_fg": "black"

    },
    "Forest": {
        "bg": "#F0F8FF", # AliceBlue
        "fg": "#2F4F4F", # DarkSlateGray
        "board_frame_bg": "#D2B48C", # Tan
        "button_bg": "#FFEBCD", # BlanchedAlmond
        "button_active_bg": "#FAEBD7", # AntiqueWhite
        "X_color": "#006400", # DarkGreen
        "O_color": "#8B4513", # SaddleBrown
        "win_highlight_bg": "#90EE90", # LightGreen
        "hint_highlight_bg": "#FFFACD", # LemonChiffon
        "text_main": "#2F4F4F",
        "text_button": "#2F4F4F",
        "accent_button_bg": "#556B2F", # DarkOliveGreen
        "accent_button_fg": "white"
    }
}

class TicTacToeLogic:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.human_symbol = 'X'
        self.ai_symbol = 'O'
        self.player1_symbol = 'X' # For PvP
        self.player2_symbol = 'O' # For PvP
        self.current_player_symbol = 'X'
        self.difficulty = "hard"
        self.game_mode = "PvAI" # PvAI or PvP
        self.scores = {"human": 0, "ai": 0, "player1": 0, "player2": 0, "draws": 0}
        self.game_over = False
        self.move_history = [] # For undo: stores (board_copy, current_player_symbol_before_move)

    def initialize_board(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.move_history = []
        return self.board

    def set_symbols_pvai(self, human_choice):
        self.human_symbol = human_choice
        self.ai_symbol = 'O' if human_choice == 'X' else 'X'

    def set_starting_player(self, starter_choice_is_human_or_p1):
        if self.game_mode == "PvAI":
            self.current_player_symbol = self.human_symbol if starter_choice_is_human_or_p1 else self.ai_symbol
        else: # PvP
            self.current_player_symbol = self.player1_symbol if starter_choice_is_human_or_p1 else self.player2_symbol
    
    def set_difficulty(self, difficulty_level):
        self.difficulty = difficulty_level

    def set_game_mode(self, mode):
        self.game_mode = mode
        if mode == "PvP": # Reset symbols for PvP clarity
            self.player1_symbol = 'X'
            self.player2_symbol = 'O'

    def get_available_moves(self):
        return [(r, c) for r, row in enumerate(self.board) for c, cell in enumerate(row) if cell == ' ']

    def check_winner(self, player_symbol_to_check): # Renamed for clarity
        b = self.board
        for r in range(3):
            if all(b[r][c] == player_symbol_to_check for c in range(3)): return [(r, c) for c in range(3)]
        for c in range(3):
            if all(b[r][c] == player_symbol_to_check for r in range(3)): return [(r, c) for r in range(3)]
        if all(b[i][i] == player_symbol_to_check for i in range(3)): return [(i, i) for i in range(3)]
        if all(b[i][2 - i] == player_symbol_to_check for i in range(3)): return [(i, 2 - i) for i in range(3)]
        return None

    def is_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def _evaluate_temp_board(self, temp_board, maximizer_s, minimizer_s):
        if self._check_winner_on_temp_board(temp_board, maximizer_s): return 1
        if self._check_winner_on_temp_board(temp_board, minimizer_s): return -1
        return 0

    def _check_winner_on_temp_board(self, tb, ps): # tb=temp_board, ps=player_symbol
        for r in range(3):
            if all(tb[r][c] == ps for c in range(3)): return True
        for c in range(3):
            if all(tb[r][c] == ps for r in range(3)): return True
        if all(tb[i][i] == ps for i in range(3)) or all(tb[i][2-i] == ps for i in range(3)): return True
        return False
    
    def _is_full_on_temp_board(self, tb):
        return all(cell != ' ' for row in tb for cell in row)

    def minimax(self, temp_board_state, depth, is_max, alpha, beta, max_s, min_s, search_limit=None):
        score_eval = self._evaluate_temp_board(temp_board_state, max_s, min_s)
        if score_eval == 1: return WIN_SCORE - depth
        if score_eval == -1: return -WIN_SCORE + depth
        if self._is_full_on_temp_board(temp_board_state): return 0
        if search_limit is not None and depth >= search_limit: return 0

        temp_avail_moves = [(r, c) for r, row in enumerate(temp_board_state) for c, cell in enumerate(row) if cell == ' ']
        
        if is_max:
            best = -math.inf
            for r, c in temp_avail_moves:
                temp_board_state[r][c] = max_s
                eval_score = self.minimax(temp_board_state, depth + 1, False, alpha, beta, max_s, min_s, search_limit)
                temp_board_state[r][c] = ' '
                best = max(best, eval_score)
                alpha = max(alpha, best)
                if beta <= alpha: break
            return best
        else:
            best = math.inf
            for r, c in temp_avail_moves:
                temp_board_state[r][c] = min_s
                eval_score = self.minimax(temp_board_state, depth + 1, True, alpha, beta, max_s, min_s, search_limit)
                temp_board_state[r][c] = ' '
                best = min(best, eval_score)
                beta = min(beta, best)
                if beta <= alpha: break
            return best

    def get_best_move(self, current_player_is_maximizer_s, opponent_is_minimizer_s, search_depth_limit=None):
        # This function finds the best move for 'current_player_is_maximizer_s'
        best_score_val = -math.inf
        best_moves_candidates = []
        available_moves_list = self.get_available_moves()

        if not available_moves_list: return None

        if len(available_moves_list) == 9 and (1,1) in available_moves_list : return (1,1) # Prioritize center for first move
        if len(available_moves_list) == 9: # if center not avail, or generally first move.
            corners = [(0,0), (0,2), (2,0), (2,2)]
            random.shuffle(corners)
            for corner_move in corners:
                if corner_move in available_moves_list: return corner_move

        temp_board_copy = [row[:] for row in self.board] # Use a copy for simulation

        for r_m, c_m in available_moves_list:
            temp_board_copy[r_m][c_m] = current_player_is_maximizer_s # AI makes a move
            # Now, evaluate this move by seeing what the opponent (minimizer) would do next.
            # So, is_maximizing for the next call to minimax is False.
            move_s = self.minimax(temp_board_copy, 0, False, -math.inf, math.inf, current_player_is_maximizer_s, opponent_is_minimizer_s, search_depth_limit)
            temp_board_copy[r_m][c_m] = ' ' # Undo the move
            
            if move_s > best_score_val:
                best_score_val = move_s
                best_moves_candidates = [(r_m, c_m)]
            elif move_s == best_score_val:
                best_moves_candidates.append((r_m, c_m))
        
        return random.choice(best_moves_candidates) if best_moves_candidates else (random.choice(available_moves_list) if available_moves_list else None)

    def get_random_move(self):
        return random.choice(self.get_available_moves()) if self.get_available_moves() else None

    def make_move(self, r_coord, c_coord, symbol_to_place):
        if self.board[r_coord][c_coord] == ' ' and not self.game_over:
            self.move_history.append({'board': copy.deepcopy(self.board), 'player': self.current_player_symbol, 'last_move':(r_coord,c_coord)})
            self.board[r_coord][c_coord] = symbol_to_place
            return True
        return False

    def switch_player(self):
        if self.game_mode == "PvAI":
            self.current_player_symbol = self.ai_symbol if self.current_player_symbol == self.human_symbol else self.human_symbol
        else: # PvP
            self.current_player_symbol = self.player2_symbol if self.current_player_symbol == self.player1_symbol else self.player1_symbol
        
    def get_ai_action(self):
        if self.difficulty == "easy": return self.get_random_move()
        if self.difficulty == "medium": return self.get_best_move(self.ai_symbol, self.human_symbol, search_depth_limit=2)
        if self.difficulty == "hard": return self.get_best_move(self.ai_symbol, self.human_symbol, search_depth_limit=None)
        return None
    
    def get_hint_for_player(self, player_s, opponent_s):
        if self.game_over: return None
        return self.get_best_move(player_s, opponent_s, search_depth_limit=None) # Hint uses max depth

    def undo_last_move(self):
        if not self.move_history: return False
        
        moves_to_revert = 0
        if self.game_mode == "PvAI":
            # If AI just moved, human player clicked Undo. Revert AI then Human move.
            # Human is current player implies AI made the last move in self.board
            if self.current_player_symbol == self.human_symbol and len(self.move_history) >= 2 :
                moves_to_revert = 2 
            # If Human just moved and wants to undo before AI (AI hasn't played yet), revert 1.
            # Or, if it's the very first move of the game by human.
            elif self.current_player_symbol == self.ai_symbol and len(self.move_history) >=1 :
                 moves_to_revert = 1
            elif len(self.move_history) >=1 : # Catch all single undo if only one move in history
                moves_to_revert = 1

        elif self.game_mode == "PvP" and len(self.move_history) >= 1:
            moves_to_revert = 1
        
        if moves_to_revert == 0 and len(self.move_history) >=1: # general fallback if no moves but history has an item.
            moves_to_revert = 1


        if moves_to_revert > 0 and moves_to_revert <= len(self.move_history) :
            restored_state = None
            for _ in range(moves_to_revert):
                 restored_state = self.move_history.pop()
            
            if restored_state:
                self.board = copy.deepcopy(restored_state['board'])
                self.current_player_symbol = restored_state['player']
                self.game_over = False # Game is no longer over if undone
                return True
        return False

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.game_logic = TicTacToeLogic()
        self.current_theme = THEMES["Classic"]

        self.font_large = ("Arial", 32, "bold") # Slightly smaller for better fit
        self.font_medium = ("Arial", 12)
        self.font_small = ("Arial", 10)

        self.cell_buttons = [[None for _ in range(3)] for _ in range(3)]
        self.hinted_cell = None
        
        self.setup_styles()
        self.create_widgets()
        self.apply_theme("Classic")
        self.reset_game_state_and_ui()

    def setup_styles(self):
        self.style = ttk.Style()
        # self.style.theme_use('clam') # Or 'alt', 'default', 'classic'
        # Overriding specific widget styles often done here if not directly in apply_theme
        # Example for all TButton: self.style.configure("TButton", ...) 
        # More granular control later within apply_theme by configuring specific styled names.


    def apply_theme(self, theme_name):
        self.current_theme = THEMES[theme_name]
        ct = self.current_theme

        self.root.configure(bg=ct["bg"])
        
        # General styles for ttk widgets based on theme's fg/bg
        self.style.configure("TLabel", background=ct["bg"], foreground=ct["text_main"], font=self.font_medium)
        self.style.configure("TFrame", background=ct["bg"])
        self.style.configure("TButton", font=self.font_small, padding=5) # General TButton
        self.style.map("TButton",
            foreground=[('!active', ct["text_button"]), ('active', ct["text_button"])],
            background=[('!active', ct["button_bg"]), ('active', ct["button_active_bg"])])

        # Specific styles that can be applied to particular buttons
        self.style.configure("Accent.TButton", font=self.font_medium, padding=6)
        self.style.map("Accent.TButton",
            foreground=[('!active', ct["accent_button_fg"]), ('active', ct["accent_button_fg"])],
            background=[('!active', ct["accent_button_bg"]), ('active', ct["button_active_bg"])]) # Active color can be different

        # For tk.Button (board cells), colors are set directly.
        # For Labels not managed by ttk, need direct config:
        if hasattr(self, 'status_label') and isinstance(self.status_label, tk.Label): # tk.Label example
             self.status_label.config(bg=ct["bg"], fg=ct["text_main"])
        if hasattr(self, 'score_label') and isinstance(self.score_label, ttk.Label): # It is ttk.Label
            self.score_label.config(background=ct["bg"], foreground=ct["text_main"])
        
        # Re-style cell buttons
        for r in range(3):
            for c in range(3):
                if self.cell_buttons[r][c]:
                    self.cell_buttons[r][c].config(bg=ct["button_bg"], activebackground=ct["button_active_bg"], fg=ct["text_button"]) # General button styling

        # Control Frame
        if hasattr(self, 'control_frame_top'):
            self.control_frame_top.config(style="TFrame")
            for child in self.control_frame_top.winfo_children():
                if isinstance(child, ttk.Label): child.config(style="TLabel")
                elif isinstance(child, ttk.OptionMenu): pass # OptionMenu is trickier, often uses OS native.

        # Refresh UI
        self.update_board_ui()
        self.update_score_ui()
        current_status_text = self.status_label.cget("text") # preserve text
        self.status_label.config(text=current_status_text) # re-apply potentially themed settings.

    def create_widgets(self):
        self.root.title("Advanced Tic-Tac-Toe")
        self.root.geometry("500x650")
        self.root.minsize(450,600)

        # Top Control Frame (Settings)
        self.control_frame_top = ttk.Frame(self.root, padding="10", style="TFrame")
        self.control_frame_top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.control_frame_top, text="Mode:", font=self.font_small, style="TLabel").pack(side=tk.LEFT, padx=(0,2))
        self.mode_var = tk.StringVar(value=self.game_logic.game_mode)
        mode_menu = ttk.OptionMenu(self.control_frame_top, self.mode_var, "PvAI", "PvAI", "PvP", command=self.on_game_settings_change)
        mode_menu.pack(side=tk.LEFT, padx=(0,10))

        self.symbol_label = ttk.Label(self.control_frame_top, text="Your Symbol (PvAI):", font=self.font_small, style="TLabel")
        self.symbol_label.pack(side=tk.LEFT, padx=(0,2))
        self.symbol_var = tk.StringVar(value=self.game_logic.human_symbol)
        self.symbol_menu = ttk.OptionMenu(self.control_frame_top, self.symbol_var, self.game_logic.human_symbol, 'X', 'O', command=self.on_game_settings_change)
        self.symbol_menu.pack(side=tk.LEFT, padx=(0,10))

        self.starter_label = ttk.Label(self.control_frame_top, text="Starts:", font=self.font_small, style="TLabel")
        self.starter_label.pack(side=tk.LEFT, padx=(0,2))
        self.starter_var = tk.StringVar(value="Human") # Default, changes text based on mode
        self.starter_menu = ttk.OptionMenu(self.control_frame_top, self.starter_var, "Human", "Human", "AI", command=self.on_game_settings_change) # Options updated dynamically
        self.starter_menu.pack(side=tk.LEFT, padx=(0,10))
        
        self.difficulty_label = ttk.Label(self.control_frame_top, text="AI Diff (PvAI):", font=self.font_small, style="TLabel")
        self.difficulty_label.pack(side=tk.LEFT, padx=(0,2))
        self.difficulty_var = tk.StringVar(value=self.game_logic.difficulty.capitalize())
        self.difficulty_menu = ttk.OptionMenu(self.control_frame_top, self.difficulty_var, self.game_logic.difficulty.capitalize(), "Easy", "Medium", "Hard", command=self.on_game_settings_change)
        self.difficulty_menu.pack(side=tk.LEFT, padx=(0,10))

        # Board Frame
        board_frame = ttk.Frame(self.root, padding="10", style="TFrame") # Themed this frame too
        board_frame.pack(expand=True, fill=tk.BOTH)

        for r in range(3):
            for c in range(3):
                button = tk.Button(board_frame, text=' ', font=self.font_large, width=3, height=1, relief=tk.GROOVE, borderwidth=2,
                                   command=lambda r_idx=r, c_idx=c: self.on_cell_click(r_idx, c_idx))
                button.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                self.cell_buttons[r][c] = button
        
        for i in range(3):
            board_frame.grid_rowconfigure(i, weight=1)
            board_frame.grid_columnconfigure(i, weight=1)

        # Status Label (Using tk.Label for more direct color control if ttk.Label is tricky with some themes)
        self.status_label = tk.Label(self.root, text="Start playing!", font=self.font_medium, anchor="center", pady=5)
        self.status_label.pack(fill=tk.X)

        # Score Label
        self.score_label = ttk.Label(self.root, text=self.get_score_text(), font=self.font_medium, anchor="center", style="TLabel")
        self.score_label.pack(fill=tk.X)
        
        # Bottom Control Frame (Actions)
        bottom_control_frame = ttk.Frame(self.root, padding="10", style="TFrame")
        bottom_control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.hint_button = ttk.Button(bottom_control_frame, text="Hint", command=self.on_hint_click, style="TButton")
        self.hint_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.undo_button = ttk.Button(bottom_control_frame, text="Undo", command=self.on_undo_click, style="TButton")
        self.undo_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        reset_button = ttk.Button(bottom_control_frame, text="New Game", command=self.reset_game_state_and_ui, style="Accent.TButton")
        reset_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        ttk.Label(self.control_frame_top, text="Theme:", font=self.font_small, style="TLabel").pack(side=tk.LEFT, padx=(0,2))
        self.theme_var = tk.StringVar(value="Classic")
        theme_menu = ttk.OptionMenu(self.control_frame_top, self.theme_var, "Classic", "Classic", "Dark", "Forest", command=self.on_theme_change)
        theme_menu.pack(side=tk.LEFT, padx=(0,5))

        self.on_game_settings_change() # Initial setup for dynamic labels based on mode

    def on_theme_change(self, theme_name):
        self.apply_theme(theme_name)
        self.update_board_ui() # Crucial to re-apply colors to board symbols etc.


    def on_game_settings_change(self, _=None):
        selected_mode = self.mode_var.get()
        self.game_logic.set_game_mode(selected_mode)

        if selected_mode == "PvAI":
            self.symbol_label.config(text="Your Symbol:")
            self.starter_label.config(text="Starts:")
            # Update starter_menu options for PvAI
            current_starter_val = self.starter_var.get()
            self.starter_menu['menu'].delete(0, 'end')
            self.starter_menu['menu'].add_command(label="Human", command=tk._setit(self.starter_var, "Human", self.on_game_settings_change))
            self.starter_menu['menu'].add_command(label="AI", command=tk._setit(self.starter_var, "AI", self.on_game_settings_change))
            if current_starter_val not in ["Human", "AI"]: self.starter_var.set("Human") # Default if prev val incompatible
            
            self.difficulty_label.config(text="AI Diff:")
            self.difficulty_menu.config(state=tk.NORMAL)
            self.symbol_menu.config(state=tk.NORMAL)
            self.hint_button.config(state=tk.NORMAL)
        else: # PvP
            self.symbol_label.config(text="P1 is X, P2 is O") # Fixed symbols for PvP
            self.starter_label.config(text="P1 Starts:")
            # Update starter_menu options for PvP
            current_starter_val = self.starter_var.get()
            self.starter_menu['menu'].delete(0, 'end')
            self.starter_menu['menu'].add_command(label="Player 1", command=tk._setit(self.starter_var, "Player 1", self.on_game_settings_change))
            self.starter_menu['menu'].add_command(label="Player 2", command=tk._setit(self.starter_var, "Player 2", self.on_game_settings_change))
            if current_starter_val not in ["Player 1", "Player 2"]: self.starter_var.set("Player 1")

            self.difficulty_label.config(text="AI N/A")
            self.difficulty_menu.config(state=tk.DISABLED)
            self.symbol_menu.config(state=tk.DISABLED)
            self.hint_button.config(state=tk.DISABLED)
        
        # Settings apply on "New Game" button or can be triggered from here for instant effect on NEXT game.
        # For now, game is reset using "New Game"

    def reset_game_state_and_ui(self):
        # Apply staged settings before resetting
        self.game_logic.set_game_mode(self.mode_var.get())
        if self.game_logic.game_mode == "PvAI":
            self.game_logic.set_symbols_pvai(self.symbol_var.get())
            self.game_logic.set_starting_player(self.starter_var.get() == "Human")
            self.game_logic.set_difficulty(self.difficulty_var.get().lower())
        else: # PvP
            self.game_logic.set_starting_player(self.starter_var.get() == "Player 1")
        
        self.game_logic.initialize_board()
        
        for r_btn in range(3):
            for c_btn in range(3):
                btn = self.cell_buttons[r_btn][c_btn]
                btn.config(text=' ', state=tk.NORMAL, relief=tk.GROOVE, bg=self.current_theme["button_bg"])
        
        self.update_board_ui() # Clear text and colors properly
        self.set_initial_status()
        self.update_score_ui()
        self.update_action_button_states()

        if self.game_logic.game_mode == "PvAI" and self.game_logic.current_player_symbol == self.game_logic.ai_symbol:
            self.root.after(500, self.trigger_ai_move)


    def set_initial_status(self):
        if self.game_logic.game_mode == "PvAI":
            if self.game_logic.current_player_symbol == self.game_logic.human_symbol:
                self.status_label.config(text=f"Your turn ({self.game_logic.human_symbol})")
            else:
                self.status_label.config(text=f"AI's turn ({self.game_logic.ai_symbol})")
        else: # PvP
             self.status_label.config(text=f"Player {self.game_logic.current_player_symbol}'s turn")

    def update_board_ui(self):
        ct = self.current_theme
        for r_cell in range(3):
            for c_cell in range(3):
                symbol = self.game_logic.board[r_cell][c_cell]
                btn = self.cell_buttons[r_cell][c_cell]
                btn.config(text=symbol)
                
                # Determine foreground color based on symbol
                fg_color = ct["text_button"] # Default
                if symbol == self.game_logic.human_symbol or symbol == self.game_logic.player1_symbol: # For PvAI or P1
                    fg_color = ct["X_color"]
                elif symbol == self.game_logic.ai_symbol or symbol == self.game_logic.player2_symbol: # For PvAI or P2
                    fg_color = ct["O_color"]
                
                btn.config(fg=fg_color) # For tk.Button, disabledforeground doesn't always play nice with theme bg

                if symbol != ' ':
                    btn.config(state=tk.DISABLED) # Standard disable
                else:
                    btn.config(state=tk.NORMAL, bg=ct["button_bg"]) # Re-enable and reset bg if cell cleared by undo


    def on_cell_click(self, r_clk, c_clk):
        if self.game_logic.game_over: return
        
        current_s = self.game_logic.current_player_symbol
        is_human_turn_pvai = (self.game_logic.game_mode == "PvAI" and current_s == self.game_logic.human_symbol)
        is_human_turn_pvp = (self.game_logic.game_mode == "PvP") # Any click in PvP is by a human

        if not (is_human_turn_pvai or is_human_turn_pvp): return

        if self.game_logic.make_move(r_clk, c_clk, current_s):
            self.update_board_ui() # Reflects the new move and disables button

            if self.check_game_over_conditions():
                self.update_action_button_states()
                return
            
            self.game_logic.switch_player()
            
            if self.game_logic.game_mode == "PvAI":
                self.status_label.config(text=f"AI's turn ({self.game_logic.ai_symbol})")
                self.root.update_idletasks()
                self.root.after(200 + random.randint(0, 300), self.trigger_ai_move)
            else: # PvP
                 self.status_label.config(text=f"Player {self.game_logic.current_player_symbol}'s turn")
        self.update_action_button_states()


    def trigger_ai_move(self):
        if self.game_logic.game_over or self.game_logic.current_player_symbol != self.game_logic.ai_symbol:
            return

        self.status_label.config(text=f"AI ({self.game_logic.ai_symbol}) is thinking...")
        self.root.update_idletasks()

        ai_move_coords = self.game_logic.get_ai_action()

        if ai_move_coords:
            r_ai, c_ai = ai_move_coords
            self.game_logic.make_move(r_ai, c_ai, self.game_logic.ai_symbol) # AI move records to history
            self.update_board_ui() # Reflects AI move

            if self.check_game_over_conditions():
                self.update_action_button_states()
                return
            
            self.game_logic.switch_player()
            self.status_label.config(text=f"Your turn ({self.game_logic.human_symbol})")
        else: # AI cannot move
             if self.game_logic.is_full(): # If board is full, could be a draw missed by prior checks
                if not self.check_game_over_conditions(): # Explicitly check if it's already a draw.
                    self.status_label.config(text="It's a Draw!")
                    self.game_logic.scores["draws"] += 1
                    self.game_logic.game_over = True
                    self.disable_all_cells_final() # Final disable, no win line
                    self.update_score_ui()
             else: # Should be rare: AI can't move on a non-full board.
                print("DEBUG: AI had no moves but board not full.") 
        self.update_action_button_states()


    def check_game_over_conditions(self):
        # Check based on the player WHO JUST MOVED (i.e., current_player_symbol before switch, or symbol on board)
        player_who_just_moved = self.game_logic.board[self.game_logic.move_history[-1]['last_move'][0]][self.game_logic.move_history[-1]['last_move'][1]] if self.game_logic.move_history else None

        winning_line = self.game_logic.check_winner(player_who_just_moved) if player_who_just_moved else None
        
        if winning_line:
            self.highlight_winning_line(winning_line)
            self.game_logic.game_over = True
            if self.game_logic.game_mode == "PvAI":
                if player_who_just_moved == self.game_logic.human_symbol:
                    self.status_label.config(text="Congratulations! You Win!")
                    self.game_logic.scores["human"] += 1
                else:
                    self.status_label.config(text=f"AI ({self.game_logic.ai_symbol}) Wins!")
                    self.game_logic.scores["ai"] += 1
            else: # PvP
                winner_num = "1 (X)" if player_who_just_moved == self.game_logic.player1_symbol else "2 (O)"
                self.status_label.config(text=f"Player {winner_num} Wins!")
                if player_who_just_moved == self.game_logic.player1_symbol: self.game_logic.scores["player1"] += 1
                else: self.game_logic.scores["player2"] += 1
            
            self.disable_all_cells_final()
            self.update_score_ui()
            return True
        elif self.game_logic.is_full():
            self.status_label.config(text="It's a Draw!")
            self.game_logic.scores["draws"] += 1
            self.game_logic.game_over = True
            self.disable_all_cells_final()
            self.update_score_ui()
            return True
        return False

    def disable_all_cells_final(self): # Called at actual game end
        for r_d in range(3):
            for c_d in range(3):
                 # We don't change already disabled for win-line highlighting.
                 # If a cell is NORMAL it means it wasn't part of win line / empty in draw.
                if self.cell_buttons[r_d][c_d]['state'] == tk.NORMAL:
                    self.cell_buttons[r_d][c_d].config(state=tk.DISABLED)
    
    def update_action_button_states(self):
        # Undo Button
        if self.game_logic.move_history and not self.game_logic.game_over :
             # In PvAI, allow undo if it's human's turn OR AI's turn but human made last physical play.
             # Simplification: allow if history exists and game not over. Logic in undo_last_move will handle PvAI turn complexity.
            self.undo_button.config(state=tk.NORMAL)
        else:
            self.undo_button.config(state=tk.DISABLED)

        # Hint Button
        is_pvai_human_turn = (self.game_logic.game_mode == "PvAI" and 
                              self.game_logic.current_player_symbol == self.game_logic.human_symbol and
                              not self.game_logic.game_over)
        self.hint_button.config(state=tk.NORMAL if is_pvai_human_turn else tk.DISABLED)

    def highlight_winning_line(self, line_coordinates):
        for r_h, c_h in line_coordinates:
            self.cell_buttons[r_h][c_h].config(bg=self.current_theme["win_highlight_bg"])

    def get_score_text(self):
        s = self.game_logic.scores
        if self.game_logic.game_mode == "PvAI":
            return f"Scores: You: {s['human']} | AI: {s['ai']} | Draws: {s['draws']}"
        else: # PvP
            return f"Scores: P1 (X): {s['player1']} | P2 (O): {s['player2']} | Draws: {s['draws']}"


    def update_score_ui(self):
        self.score_label.config(text=self.get_score_text())

    def on_undo_click(self):
        if self.game_logic.undo_last_move():
            self.update_board_ui() # This should re-enable cells and reset their text/colors based on restored board
            self.set_initial_status() # Status based on whose turn it is now
             # Clear any win highlights or previous hint highlights
            for r_btn in range(3):
                for c_btn in range(3):
                    if self.game_logic.board[r_btn][c_btn] == ' ': # if cell is now empty
                         self.cell_buttons[r_btn][c_btn].config(bg=self.current_theme["button_bg"], state=tk.NORMAL)
                    # else, update_board_ui would have set X/O colors, no explicit bg for them.
            self.status_label.config(text=f"{self.game_logic.current_player_symbol}'s turn.") # Generic status for undo
        
        self.update_action_button_states()
        if self.game_logic.game_mode == "PvAI" and self.game_logic.current_player_symbol == self.game_logic.ai_symbol:
            # Important: Do NOT auto-trigger AI move after undo. Player should re-evaluate.
            # If after undo, it IS AI's turn logically based on history, player can choose to pass (if implemented) or New Game.
            # For TicTacToe, AI turn would mean human needs to undo AGAIN to get to their previous choice.
            # The logic of undo_last_move in PvAI already tries to revert player to *their* previous decision point.
            self.status_label.config(text=f"Your turn ({self.game_logic.human_symbol})")


    def on_hint_click(self):
        if self.game_logic.game_mode == "PvAI" and self.game_logic.current_player_symbol == self.game_logic.human_symbol:
            hint_move = self.game_logic.get_hint_for_player(self.game_logic.human_symbol, self.game_logic.ai_symbol)
            if hint_move:
                r_h, c_h = hint_move
                if self.hinted_cell and self.game_logic.board[self.hinted_cell[0]][self.hinted_cell[1]] == ' ': # Clear previous hint if cell still empty
                    self.cell_buttons[self.hinted_cell[0]][self.hinted_cell[1]].config(bg=self.current_theme["button_bg"])

                self.hinted_cell = (r_h, c_h)
                original_bg = self.cell_buttons[r_h][c_h].cget("background")
                self.cell_buttons[r_h][c_h].config(bg=self.current_theme["hint_highlight_bg"])
                # Revert after a delay, only if the cell hasn't been clicked
                def revert_hint_highlight():
                    if self.hinted_cell == (r_h, c_h) and self.game_logic.board[r_h][c_h] == ' ': # If cell is still the hinted one and empty
                        self.cell_buttons[r_h][c_h].config(bg=self.current_theme["button_bg"]) # Use theme specific original
                        if self.hinted_cell == (r_h, c_h) : self.hinted_cell = None # Clear if it's this specific hint we're reverting

                self.root.after(1500, revert_hint_highlight)


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()