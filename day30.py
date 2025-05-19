import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import os
import json # For saving/loading config
import wandb # Weights & Biases
import optuna # Hyperparameter optimization

# --- Configuration ---
# Encapsulate in a function to allow Optuna to modify it
def get_config():
    return {
        "data_path": 'essays.csv',
        "transformer_model_name": 'bert-base-uncased', # Can be 'roberta-base', etc.
        "max_len": 512,
        "batch_size": 8,
        "epochs": 5, # Keep this moderate for Optuna trials, can be higher for final run
        "learning_rate": 2e-5,
        "optimizer_eps": 1e-8, # Epsilon for AdamW
        "warmup_steps_ratio": 0.1, # Ratio of total steps for warmup
        "patience": 3,
        "seed": 42,
        "model_save_path": "best_essay_grading_model.pth",
        "config_save_path": "best_model_config.json",
        "regression_head_layers": 1, # Number of linear layers after BERT (1 means original, >1 means MLP)
        "regression_head_dropout": 0.1,
        "wandb_project": "EssayGrading", # Your W&B project name
        "wandb_entity": None, # Your W&B username or team (optional, W&B will use default)
        "run_optuna": False, # Set to True to run hyperparameter optimization
        "optuna_trials": 10, # Number of Optuna trials
    }

CONFIG = get_config()

# --- Reproducibility ---
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Step 1: Load and Explore the Dataset ---
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Generating dummy data.")
        num_samples = 200 # Increased for better training stability with dummy data
        dummy_essays = [f"This is sample essay number {i}. It discusses various important topics and provides some analysis." for i in range(num_samples)]
        dummy_scores = np.random.uniform(1, 10, size=num_samples).round(1)
        df = pd.DataFrame({'Essay': dummy_essays, 'Overall': dummy_scores})
    else:
        df = pd.read_csv(file_path, encoding='latin1')
        if 'essay' in df.columns and 'Essay' not in df.columns: df.rename(columns={'essay': 'Essay'}, inplace=True)
        if 'final_score' in df.columns and 'Overall' not in df.columns: df.rename(columns={'final_score': 'Overall'}, inplace=True)

    print("Dataset Sample:")
    print(df.head())
    if 'Essay' not in df.columns or 'Overall' not in df.columns:
        raise ValueError("Dataset must contain 'Essay' and 'Overall' columns.")
    df = df[['Essay', 'Overall']]
    df.dropna(subset=['Essay', 'Overall'], inplace=True)
    df['Overall'] = pd.to_numeric(df['Overall'], errors='coerce')
    df.dropna(subset=['Overall'], inplace=True)
    print(f"Loaded {len(df)} samples.")
    return df

# --- Step 2: Preprocess the Data ---
class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_len):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        essay = str(self.essays[index])
        score = self.scores[index]
        encoding = self.tokenizer.encode_plus(
            essay, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }

# --- Step 3: Build the Model ---
class EssayGradingModel(torch.nn.Module):
    def __init__(self, transformer_model_name, regression_head_layers=1, regression_head_dropout=0.1):
        super(EssayGradingModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.config = self.transformer.config
        
        layers = []
        input_size = self.config.hidden_size
        for i in range(regression_head_layers - 1):
            layers.append(torch.nn.Linear(input_size, self.config.hidden_size // 2))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(regression_head_dropout))
            input_size = self.config.hidden_size // 2
        
        layers.append(torch.nn.Linear(input_size, 1))
        self.regressor = torch.nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] # Use [CLS] token embedding
        score = self.regressor(cls_output)
        return score.squeeze(-1)

# --- Step 4: Train the Model ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, current_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, scores)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if wandb.run: # Log batch loss and learning rate to W&B
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        if batch_idx % 50 == 0:
            print(f"  Epoch {current_epoch+1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
    return total_loss / len(data_loader)

# --- Step 5: Evaluate the Model ---
def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    predictions_list = []
    true_scores_list = []
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, scores)
            total_loss += loss.item()
            predictions_list.extend(outputs.cpu().numpy())
            true_scores_list.extend(scores.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    mse = mean_squared_error(true_scores_list, predictions_list)
    mae = mean_absolute_error(true_scores_list, predictions_list)
    r2 = r2_score(true_scores_list, predictions_list)
    pearson_corr, _ = pearsonr(true_scores_list, predictions_list) if np.std(true_scores_list) > 0 and np.std(predictions_list) > 0 else (0, 1.0)
    
    return avg_loss, mse, mae, r2, pearson_corr, true_scores_list, predictions_list

# --- Training Loop ---
def run_training_session(current_config, train_loader, val_loader, trial=None): # trial for Optuna
    set_seed(current_config["seed"])
    
    model = EssayGradingModel(
        current_config["transformer_model_name"],
        current_config["regression_head_layers"],
        current_config["regression_head_dropout"]
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=current_config["learning_rate"], eps=current_config["optimizer_eps"])
    loss_fn = torch.nn.MSELoss()
    
    total_steps = len(train_loader) * current_config["epochs"]
    num_warmup_steps = int(total_steps * current_config["warmup_steps_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses_history, val_losses_history = [], []

    # Initialize W&B if not an Optuna trial or if it's the best Optuna trial
    if wandb.run is None and not trial: # Only init if not already running (e.g. from Optuna)
        wandb.init(
            project=current_config["wandb_project"],
            entity=current_config["wandb_entity"],
            config=current_config,
            reinit=True # Allows re-init if called multiple times in a script
        )
        wandb.watch(model, log="all", log_freq=100) # Log gradients and parameters

    for epoch in range(current_config["epochs"]):
        print(f'Epoch {epoch+1}/{current_config["epochs"]}')
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, epoch, current_config["epochs"])
        train_losses_history.append(train_loss)
        
        val_loss, mse, mae, r2, pearson, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        val_losses_history.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'  Val MSE: {mse:.4f}, Val MAE: {mae:.4f}, Val R2: {r2:.4f}, Val Pearson: {pearson:.4f}')

        if wandb.run:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": mse,
                "val_mae": mae,
                "val_r2": r2,
                "val_pearson": pearson
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not trial: # Don't save model for every optuna trial, only for the final best one
                torch.save(model.state_dict(), current_config["model_save_path"])
                with open(current_config["config_save_path"], 'w') as f:
                    json.dump(current_config, f, indent=4)
                print(f"  New best model and config saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Val loss did not improve. Counter: {epochs_no_improve}/{current_config['patience']}")

        if epochs_no_improve >= current_config["patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
        if trial: # For Optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                wandb.finish(quiet=True) # End W&B run for pruned trial
                raise optuna.exceptions.TrialPruned()
                
    if wandb.run and not trial: # If it's the main run, not optuna trial that might be pruned
         wandb.finish()
         
    return best_val_loss, train_losses_history, val_losses_history, model # Return model for Optuna best trial case

# --- Optuna Objective Function ---
def optuna_objective(trial):
    # Suggest hyperparameters
    current_config = get_config() # Start with base config
    current_config["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    current_config["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16])
    current_config["regression_head_layers"] = trial.suggest_int("regression_head_layers", 1, 3)
    current_config["regression_head_dropout"] = trial.suggest_float("regression_head_dropout", 0.0, 0.5)
    current_config["warmup_steps_ratio"] = trial.suggest_float("warmup_steps_ratio", 0.0, 0.2)
    # Can add transformer model choice too:
    # current_config["transformer_model_name"] = trial.suggest_categorical("transformer_model_name", ['bert-base-uncased', 'roberta-base'])
    
    # Set a unique run name for W&B for this trial
    run_name = f"optuna_trial_{trial.number}_{current_config['transformer_model_name']}_lr{current_config['learning_rate']:.2e}_bs{current_config['batch_size']}"
    
    # Initialize W&B for this trial
    wandb.init(
        project=current_config["wandb_project"],
        entity=current_config["wandb_entity"],
        config=current_config,
        name=run_name,
        group="Optuna Hyperparameter Search", # Group trials together
        reinit=True, # Important for Optuna
        job_type="hyperparameter_tuning"
    )
    
    # --- Data Loading and Preparation ---
    # This part is repeated for each trial, consider optimizing if dataset loading is very slow
    # For most text datasets, this is fine.
    global df_main, tokenizer_main # Use globally loaded df and tokenizer to save time
    
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        df_main['Essay'].values, df_main['Overall'].values, test_size=0.2, random_state=current_config["seed"]
    )
    train_dataset = EssayDataset(train_texts, train_scores, tokenizer_main, current_config["max_len"])
    val_dataset = EssayDataset(val_texts, val_scores, tokenizer_main, current_config["max_len"])
    train_loader = DataLoader(train_dataset, batch_size=current_config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=current_config["batch_size"], num_workers=2, pin_memory=True)
    
    try:
        best_val_loss, _, _, _ = run_training_session(current_config, train_loader, val_loader, trial)
    except optuna.exceptions.TrialPruned:
        wandb.log({"status": "pruned"})
        wandb.finish(quiet=True)
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        wandb.log({"status": "failed", "error": str(e)})
        wandb.finish(quiet=True)
        return float('inf') # Return high loss for failed trials

    wandb.log({"status": "completed", "final_val_loss_for_optuna": best_val_loss})
    wandb.finish(quiet=True) # Finish W&B run for this trial
    return best_val_loss


# --- Step 6: Test the Model ---
def test_model_on_sample(test_text, model_path, config_path):
    print("\n--- Testing Model on Sample ---")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Error: Model or config file not found. Cannot test.")
        print(f"Looked for model: {model_path}, config: {config_path}")
        return None

    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    
    local_tokenizer = AutoTokenizer.from_pretrained(loaded_config["transformer_model_name"])
    test_model = EssayGradingModel(
        loaded_config["transformer_model_name"],
        loaded_config["regression_head_layers"],
        loaded_config["regression_head_dropout"]
    ).to(device)
    test_model.load_state_dict(torch.load(model_path, map_location=device))
    test_model.eval()
    
    encoding = local_tokenizer.encode_plus(
        test_text, add_special_tokens=True, max_length=loaded_config["max_len"], 
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        predicted_score = test_model(input_ids, attention_mask).item()
    print(f"Essay: \"{test_text}\"")
    print(f"Predicted Score: {predicted_score:.2f} (using model: {loaded_config['transformer_model_name']})")
    return predicted_score

# --- Step 7: Visualize Training Performance & Error Analysis ---
def plot_performance_and_errors(train_losses, val_losses, true_scores, pred_scores, all_texts_for_error_analysis=None):
    print("\n--- Visualizing Performance and Errors ---")
    plt.figure(figsize=(18, 6))

    # Plotting training and validation loss
    plt.subplot(1, 3, 1)
    epochs_ran = range(1, len(train_losses) + 1)
    plt.plot(epochs_ran, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs_ran, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # Plotting actual vs. predicted scores
    plt.subplot(1, 3, 2)
    plt.scatter(true_scores, pred_scores, alpha=0.6, edgecolors='w', linewidth=0.5)
    min_val = min(min(true_scores, default=0), min(pred_scores, default=0))
    max_val = max(max(true_scores, default=10), max(pred_scores, default=10))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Correlation')
    plt.title('Actual vs. Predicted Scores'); plt.xlabel('Actual Scores'); plt.ylabel('Predicted Scores')
    plt.legend(); plt.grid(True)

    # Plotting residuals
    plt.subplot(1, 3, 3)
    residuals = np.array(true_scores) - np.array(pred_scores)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals (Actual - Predicted)'); plt.xlabel('Residual'); plt.ylabel('Frequency'); plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    if wandb.run: # Log plots to W&B
        wandb.log({"performance_plots": plt})

    # Show N worst predictions
    if all_texts_for_error_analysis and true_scores and pred_scores:
        errors = np.abs(residuals)
        sorted_indices = np.argsort(errors)[::-1] # Sort by error descending
        print("\n--- Top 5 Worst Predictions (Highest Absolute Error) ---")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"Essay Text (snippet): {all_texts_for_error_analysis[idx][:100]}...") # Show a snippet
            print(f"  Actual Score: {true_scores[idx]:.2f}, Predicted Score: {pred_scores[idx]:.2f}, Error: {residuals[idx]:.2f}")


# --- Main Execution ---
df_main = None # Global for Optuna to reuse
tokenizer_main = None # Global for Optuna to reuse

if __name__ == "__main__":
    CONFIG = get_config() # Load default config
    set_seed(CONFIG["seed"])
    print(f"Using device: {device}")

    df_main = load_data(CONFIG["data_path"])
    tokenizer_main = AutoTokenizer.from_pretrained(CONFIG["transformer_model_name"])

    # Prepare data loaders (will be overridden by Optuna if run)
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        df_main['Essay'].values, df_main['Overall'].values, test_size=0.2, random_state=CONFIG["seed"]
    )
    val_texts_for_error_analysis = val_texts # Keep original texts for error display

    train_dataset = EssayDataset(train_texts, train_scores, tokenizer_main, CONFIG["max_len"])
    val_dataset = EssayDataset(val_texts, val_scores, tokenizer_main, CONFIG["max_len"])
    
    # Optuna Hyperparameter Search
    if CONFIG["run_optuna"]:
        print("\n--- Starting Optuna Hyperparameter Search ---")
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(optuna_objective, n_trials=CONFIG["optuna_trials"], timeout=1800) # Added timeout e.g. 30min

        print("\nOptuna Study Statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Best trial: Value (Val Loss): {study.best_trial.value}")
        print("  Best Prams: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
            CONFIG[key] = value # Update global CONFIG with best params
        
        # Log best trial params to W&B if a separate summary run is desired
        # For now, each trial logs itself. The best model is NOT automatically retrained/saved here.
        # You'd typically take best_params and run a final training session.
        print("\n--- Optuna search finished. Best parameters found. ---")
        print("--- You should now re-run the script with `run_optuna = False` ---")
        print("--- and manually update CONFIG with these best parameters, or ---")
        print("--- implement logic to automatically use them for a final training run. ---")
        # For simplicity, we'll just print them. A more robust pipeline would use them.
        # If you want to directly use them:
        # CONFIG["run_optuna"] = False # Prevent re-running optuna
        # And the script would continue to train with these new CONFIG values.

    # Proceed with training using either default or Optuna-found (if manually updated) CONFIG
    if not CONFIG["run_optuna"]: # Run a single training session
        print(f"\n--- Starting Single Training Session with Config: ---")
        for key, value in CONFIG.items(): print(f"  {key}: {value}")

        # Recreate DataLoaders with potentially updated batch_size from Optuna
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], num_workers=2, pin_memory=True)

        # Initialize W&B for the main run (if not already initialized by Optuna trial logic)
        if wandb.run is None: # Ensure W&B is initialized for the final/single run
            wandb.init(
                project=CONFIG["wandb_project"],
                entity=CONFIG["wandb_entity"],
                config=CONFIG,
                name=f"final_run_{CONFIG['transformer_model_name']}",
                job_type="training"
            )
            # wandb.watch(model) # Model isn't defined yet here, moved to run_training_session

        best_val_loss, train_history, val_history, trained_model = run_training_session(CONFIG, train_loader, val_loader)
        
        # Evaluate the best model (loaded from disk) on the validation set
        print("\n--- Evaluating Best Saved Model on Validation Set ---")
        if os.path.exists(CONFIG["model_save_path"]):
            best_model = EssayGradingModel(
                CONFIG["transformer_model_name"],
                CONFIG["regression_head_layers"],
                CONFIG["regression_head_dropout"]
            ).to(device)
            best_model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
            
            val_loss, mse, mae, r2, pearson, true_val_scores, pred_val_scores = evaluate_model(
                best_model, val_loader, torch.nn.MSELoss(), device
            )
            print(f'Best Model Val Loss: {val_loss:.4f}')
            print(f'  Val MSE: {mse:.4f}, Val MAE: {mae:.4f}, Val R2: {r2:.4f}, Val Pearson: {pearson:.4f}')

            if wandb.run:
                wandb.summary["best_val_loss_final_model"] = val_loss
                wandb.summary["best_val_mse_final_model"] = mse
                wandb.summary["best_val_mae_final_model"] = mae
                wandb.summary["best_val_r2_final_model"] = r2
                wandb.summary["best_val_pearson_final_model"] = pearson
            
            if train_history and val_history:
                plot_performance_and_errors(train_history, val_history, true_val_scores, pred_val_scores, val_texts_for_error_analysis)
        else:
            print(f"Model file {CONFIG['model_save_path']} not found. Cannot evaluate or plot for final model.")
        
        # Test with a sample essay using the saved best model and its config
        test_essay_good = "This essay presents a comprehensive and insightful analysis of the assigned topic. The arguments are well-structured, supported by strong evidence, and articulated with clarity and precision. The author demonstrates a profound understanding of the subject matter and effectively engages with complex ideas."
        test_model_on_sample(test_essay_good, CONFIG["model_save_path"], CONFIG["config_save_path"])
        
        test_essay_bad = "the essay is bad. it dont make no sense. i tryed but words are hard. not good structure."
        test_model_on_sample(test_essay_bad, CONFIG["model_save_path"], CONFIG["config_save_path"])

        if wandb.run: # Ensure W&B run is finished if it was started for the single run
            wandb.finish()
            
    elif CONFIG["run_optuna"]:
        print("\nOptuna study complete. To train the best model, re-run with 'run_optuna: False' and update CONFIG with the best parameters found.")