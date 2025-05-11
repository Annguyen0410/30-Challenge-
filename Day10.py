import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    # Clip input to prevent overflow/underflow with np.exp
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

def sigmoid_derivative(s): # s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(r): # r = relu(x)
    # The derivative is 1 if r > 0, and 0 otherwise.
    return (r > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(t): # t = tanh(x)
    return 1 - t**2

def linear(x):
    return x

def linear_derivative(x): # x = linear(x)
    return np.ones_like(x)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative)
}

# Loss functions
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12 # Small constant to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

LOSS_FUNCTIONS = {
    'mse': mean_squared_error,
    'bce': binary_cross_entropy
}

class NeuralNetwork:
    def __init__(self, layer_dims, hidden_activation='relu', output_activation='sigmoid',
                 loss_function='bce', weight_init_scheme='auto', l2_lambda=0.0):

        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported hidden activation: {hidden_activation}")
        if output_activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        if loss_function not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1 # Number of transformations (weight sets)
        self.loss_func = LOSS_FUNCTIONS[loss_function]
        self.loss_choice = loss_function
        self.output_activation_name = output_activation # Store name for special handling
        self.l2_lambda = l2_lambda

        self.weights = []
        self.biases = []
        self.activation_funcs = []
        self.activation_derivatives = []
        
        self._cache = {} # For storing A_vals (activations) and Z_vals (pre-activations)

        for i in range(self.num_layers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i+1]
            
            current_weight_init_scheme = weight_init_scheme
            current_activation_for_layer_name = hidden_activation if i < self.num_layers -1 else output_activation
            
            if weight_init_scheme == 'auto':
                if current_activation_for_layer_name == 'relu':
                    current_weight_init_scheme = 'he'
                elif current_activation_for_layer_name in ['sigmoid', 'tanh', 'linear']:
                    current_weight_init_scheme = 'xavier'
                else: 
                    current_weight_init_scheme = 'xavier'


            self.weights.append(self._initialize_weights(fan_in, fan_out, current_weight_init_scheme))
            self.biases.append(np.zeros((1, fan_out)))

            if i < self.num_layers - 1: # Hidden layer
                act_func, act_deriv = ACTIVATIONS[hidden_activation]
            else: # Output layer
                act_func, act_deriv = ACTIVATIONS[output_activation]
            self.activation_funcs.append(act_func)
            self.activation_derivatives.append(act_deriv)

    def _initialize_weights(self, fan_in, fan_out, scheme):
        if scheme == 'xavier':
            limit = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif scheme == 'he':
            return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
        elif scheme == 'random_small': 
            return np.random.randn(fan_in, fan_out) * 0.01
        else: 
            print(f"Warning: Unsupported weight initialization scheme '{scheme}'. Defaulting to 'xavier'.")
            limit = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))


    def forward(self, X):
        A = X
        A_vals = [X] 
        Z_vals = []  

        for i in range(self.num_layers):
            Z_next = np.dot(A, self.weights[i]) + self.biases[i]
            A_next = self.activation_funcs[i](Z_next)
            Z_vals.append(Z_next)
            A_vals.append(A_next)
            A = A_next
        
        self._cache['A_vals'] = A_vals
        self._cache['Z_vals'] = Z_vals 
        return A 

    def backward(self, X_batch_not_used, y_true, learning_rate):
        m = y_true.shape[0] 
        A_vals = self._cache['A_vals']
        final_A = A_vals[-1] 

        if self.loss_choice == 'bce' and self.output_activation_name == 'sigmoid':
            dZL = (final_A - y_true) / m
        else:
            if self.loss_choice == 'mse':
                dAL_avg = 2 * (final_A - y_true) / m
            elif self.loss_choice == 'bce':
                epsilon = 1e-12
                clipped_AL = np.clip(final_A, epsilon, 1. - epsilon)
                dAL_avg = -( (y_true / clipped_AL) - ((1 - y_true) / (1 - clipped_AL)) ) / m
            else:
                raise NotImplementedError("Loss derivative not specified for this combination.")
            
            dZL = dAL_avg * self.activation_derivatives[-1](final_A)

        dZ_current = dZL 

        for l in reversed(range(self.num_layers)):
            A_prev = A_vals[l] 
            
            dW_l = np.dot(A_prev.T, dZ_current) + (self.l2_lambda / m) * self.weights[l]
            db_l = np.sum(dZ_current, axis=0, keepdims=True)
            
            self.weights[l] -= learning_rate * dW_l
            self.biases[l] -= learning_rate * db_l

            if l > 0: 
                dA_prev = np.dot(dZ_current, self.weights[l].T)
                dZ_current = dA_prev * self.activation_derivatives[l-1](A_vals[l])


    def train(self, X_train, y_train, epochs, learning_rate, batch_size=None,
              verbose=True, print_every=100, learning_rate_decay_rate=0.0,
              validation_data=None, early_stopping_patience=None):

        initial_learning_rate = learning_rate
        num_samples = X_train.shape[0]

        if batch_size is None or batch_size <= 0 or batch_size > num_samples :
            batch_size = num_samples
        
        history = {'loss': [], 'accuracy': [], 'total_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        best_val_loss = float('inf')
        epochs_no_improve = 0 # Counts number of validation checks with no improvement

        for epoch in range(epochs):
            current_lr = initial_learning_rate
            if learning_rate_decay_rate > 0:
                current_lr = initial_learning_rate / (1 + learning_rate_decay_rate * epoch)

            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.forward(X_batch) 
                self.backward(X_batch, y_batch, current_lr) 

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                train_eval = self.evaluate(X_train, y_train, batch_size=batch_size, is_training_eval=True)
                data_loss = train_eval['loss']
                total_loss = train_eval['total_loss']
                accuracy = train_eval.get('accuracy')

                history['loss'].append(data_loss)
                history['total_loss'].append(total_loss)
                if accuracy is not None: history['accuracy'].append(accuracy)

                log_message = f"Epoch {epoch}, Train Loss: {data_loss:.4f}, Train Total Loss: {total_loss:.4f}, LR: {current_lr:.6f}"
                if accuracy is not None:
                    log_message += f", Train Acc: {accuracy:.4f}"
                
                if validation_data:
                    X_val, y_val = validation_data
                    val_eval = self.evaluate(X_val, y_val, batch_size=batch_size)
                    val_loss = val_eval['loss'] 
                    val_accuracy = val_eval.get('accuracy')

                    history['val_loss'].append(val_loss)
                    if val_accuracy is not None: history['val_accuracy'].append(val_accuracy)
                    
                    log_message += f" -- Val Loss: {val_loss:.4f}"
                    if val_accuracy is not None:
                        log_message += f", Val Acc: {val_accuracy:.4f}"

                    if early_stopping_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        
                        if epochs_no_improve >= early_stopping_patience:
                            print(log_message) # Print current epoch's stats before stopping message
                            print(f"\nEarly stopping triggered at epoch {epoch}. No improvement in validation loss for {early_stopping_patience} consecutive validation checks (evaluated every {print_every} epochs).")
                            return history
                print(log_message)
        return history


    def predict(self, X, batch_size=None):
        if batch_size is None or batch_size <=0 or batch_size >= X.shape[0]:
             return self.forward(X)
        
        num_samples = X.shape[0]
        outputs = []
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            outputs.append(self.forward(X_batch))
        return np.vstack(outputs)


    def evaluate(self, X_test, y_test, batch_size=None, is_training_eval=False):
        output = self.predict(X_test, batch_size=batch_size)
        data_loss = self.loss_func(y_test, output)
        
        l2_cost_term_value = 0
        if self.l2_lambda > 0:
            current_set_num_samples = X_test.shape[0]
            if current_set_num_samples == 0: current_set_num_samples = 1 

            sum_sq_weights = 0
            for W in self.weights:
                sum_sq_weights += np.sum(np.square(W))
            l2_cost_term_value = (self.l2_lambda / (2 * current_set_num_samples)) * sum_sq_weights
        
        total_loss = data_loss + l2_cost_term_value
        
        metrics = {'loss': data_loss, 'total_loss': total_loss}
        if self.output_activation_name == 'sigmoid' and y_test.ndim == 2 and y_test.shape[1] == 1: 
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y_test)
            metrics['accuracy'] = accuracy
        return metrics

    def summary(self):
        print("Neural Network Summary:")
        print("---------------------------------------------------------------------------")
        print(f"Layer Dims: {self.layer_dims}")
        print("---------------------------------------------------------------------------")
        print(f"{'Layer (type)':<20} {'Activation':<15} {'Output Shape':<15} {'Param #':<10}")
        print("===========================================================================")
        total_params = 0
        
        # Input Layer (conceptual)
        # Corrected line: str() added around the tuple for formatting
        print(f"{'Input':<20} {'N/A':<15} {str((None, self.layer_dims[0])):<15} {'0':<10}")

        for i in range(self.num_layers):
            layer_name = f"Dense_{i+1}"
            W_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            layer_params = np.prod(W_shape) + np.prod(b_shape)
            total_params += layer_params
            
            activation_name = self.activation_funcs[i].__name__
            # output_shape_str is already a string due to f-string evaluation
            output_shape_str = f"{(None, self.layer_dims[i+1])}" 

            print(f"{layer_name:<20} {activation_name:<15} {output_shape_str:<15} {layer_params:<10}")
        print("===========================================================================")
        print(f"Total Trainable Parameters: {total_params}")
        print(f"Loss Function: {self.loss_choice.upper()}")
        print(f"L2 Regularization Lambda: {self.l2_lambda}")
        print("---------------------------------------------------------------------------")


# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

X_train_xor, y_train_xor = X_xor, y_xor

nn_xor = NeuralNetwork(layer_dims=[2, 8, 1], 
                       hidden_activation='tanh',      
                       output_activation='sigmoid',   
                       loss_function='bce',           
                       weight_init_scheme='xavier',   
                       l2_lambda=0.001)               

nn_xor.summary()

print("\nTraining for XOR problem:")
history = nn_xor.train(X_train_xor, y_train_xor,
                       epochs=10000,
                       learning_rate=0.05,
                       batch_size=4, 
                       verbose=True,
                       print_every=1000,
                       learning_rate_decay_rate=0.001,
                       validation_data=(X_train_xor, y_train_xor), 
                       # Patience is number of check intervals (print_every)
                       early_stopping_patience=500 # Stop if val_loss doesn't improve for 500 validation checks.
                       )


print("\nTesting the trained XOR neural network:")
test_results = nn_xor.evaluate(X_xor, y_xor)
print(f"Test Loss: {test_results['loss']:.4f}, Test Accuracy: {test_results['accuracy']:.4f}")

for i in range(len(X_xor)):
    input_sample = X_xor[i:i+1] 
    prediction_raw = nn_xor.predict(input_sample)
    predicted_value = prediction_raw[0,0]
    rounded_prediction = int(predicted_value > 0.5)
    print(f"Input: {X_xor[i]}, Predicted Output: {predicted_value:.4f} (Rounded: {rounded_prediction}), Actual Output: {y_xor[i,0]}")

# Example for a regression task (simple linear data)
X_reg = np.array([[x] for x in np.linspace(-5, 5, 50)], dtype=np.float32)
y_reg = np.array([[2*x[0] + 1 + np.random.randn()*0.5] for x in X_reg], dtype=np.float32) 

nn_reg = NeuralNetwork(layer_dims=[1, 10, 1], 
                       hidden_activation='relu',
                       output_activation='linear', 
                       loss_function='mse',        
                       weight_init_scheme='he',    
                       l2_lambda=0.01)

nn_reg.summary()
print("\nTraining for Regression problem:")
reg_history = nn_reg.train(X_reg, y_reg, epochs=500, learning_rate=0.01, batch_size=10, print_every=100)

print("\nTesting the trained Regression neural network (showing a few predictions):")
test_points = np.array([[-4.0], [0.0], [4.0]], dtype=np.float32)
predictions = nn_reg.predict(test_points)
for i in range(len(test_points)):
    print(f"Input: {test_points[i,0]:.1f}, Predicted Output: {predictions[i,0]:.4f}, Actual (approx): {2*test_points[i,0]+1:.4f}")