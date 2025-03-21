from utils.paths import TMP_DIR
from utils.paths import ML_FLOW_DIR
from autoregression.models.DirMultiLSTM import build_direct_lstm_model_simple
from autoregression.models.RecurLSTM import build_recursive_lstm_simple
from autoregression.models.RefineLSTM import build_refinement_lstm_model
from autoregression.models.seq2seq_lstm import build_and_compile_seq2seq_lstm
from autoregression.utils_ar import prepare_direct_lstm_data 
from autoregression.utils_ar import prepare_recursive_lstm_data 
from autoregression.utils_ar import n_from_recursive
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import os
import matplotlib.pyplot as plt
from evaluation import evalresu as er
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from datetime import datetime

 
def train_and_save_model(model, X_train, y_train, X_val, y_val,
                         name: str,
                         epochs=30, batch_size=32,
                         patience_early_stopping=10, patience_lr_reduction=5, min_lr=1e-6):
    """
    Train LSTM model and save the best model based on validation loss. 
    Includes EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

    Parameters:
    - model: The model to train.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - name: The name for saving the trained model.
    - epochs: Number of training epochs.
    - batch_size: The batch size for training.
    - patience_early_stopping: Number of epochs to wait before stopping training if no improvement.
    - patience_lr_reduction: Number of epochs to wait before reducing the learning rate.
    - min_lr: The minimum learning rate during learning rate reduction.

    Returns:
    - trained model, training history
    """
    save_path = 'models/saved/' + name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Callback: Save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        save_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    # Callback: Stop training if validation loss does not improve (early stopping)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience_early_stopping, verbose=1, restore_best_weights=True
    )
    
    # Callback: Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience_lr_reduction, 
        min_lr=min_lr, verbose=1
    )
    
    # Fit the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Optionally return the trained model and training history
    return model, history



def plot_training_history(history, title="Training History"):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

        
mlflow.set_experiment("Recursive_LSTM_Experiments")  # Set experiment for Recursive LSTM
mlflow.set_experiment("Seq2Seq_LSTM_Experiments")  # Set experiment for Seq2Seq LSTM
mlflow.set_experiment("Hybrid_LSTM_Experiments")  # Set experiment for Hybrid LSTM

# Set up experiments for each approach
mlflow.set_experiment("Direct_LSTM_Experiments")  # Set experiment for Direct LSTM

data = joblib.load(TMP_DIR / 'output1_smoothed.pkl')
data = joblib.load(TMP_DIR / 'output1_smoothed_RTS.pkl')

n_past_values=100
n_future_values=20
input_shape=(n_past_values,1)

X_train_dir, y_train_dir,X_val_dir,y_val_dir,X_test_dir,y_test_dir = prepare_direct_lstm_data(data,n_past_values,n_future_values)
X_train_rec, y_train_rec,X_val_rec,y_val_rec,X_test_rec,y_test_rec = prepare_recursive_lstm_data(data,n_past_values)
rec_model=build_recursive_lstm(n_past_values,64)
rec_history = train_and_save_model(rec_model,X_train=X_train_rec, y_train=y_train_rec, X_val=X_val_rec, y_val=y_val_rec,name='simple_rec.keras',epochs=50)
plot_training_history(rec_history[1])


rec_pred_1=rec_model.predict(X_test_rec)
er.evaluate_n_values(rec_pred_1,y_test_rec,1)
from utils.paths import LSTM_MODELS_DIR
from tensorflow.keras.models import load_model
model_rec_simple_trained=load_model(LSTM_MODELS_DIR/'simple_rec.keras')

rec_model=load
n_predicted_rows=100
rec_pred_n=n_from_recursive(model_rec_simple_trained,input_seq=X_test_rec,length_of_sequence=5000)
arr=np.array(rec_pred_n)
hyb_train=dir_model.predict()
er.evaluate_n_values(y_test_dir[:,-1],dir_pred[:,-1],n=20)
er.evaluate_full_sequence(y_test_dir,dir_pred,2)

er.evaluate_n_values(y_test_dir[:n_predicted_rows,-1],arr[:,-1],20)
er.evaluate_full_sequence(y_test_dir,arr,1)

input_shape=()
seq2seq_model=build_and_compile_seq2seq_lstm(input_shape=input_shape)
seq2seq_model.summary()
sqe2sqe_history=train_and_save_model(seq2seq_model, X_train=X_train_dir, y_train=y_train_dir, X_val=X_val_dir, y_val=y_val_dir,name='simple_seq_2_seq.keras',epochs=50)
plot_training_history(sqe2sqe_history[1])
seq2seq_pred=np.squeeze(seq2seq_model.predict(X_test_dir))
er.evaluate_n_values(y_test_dir[:,-1],seq2seq_pred[:,-1],n=20)
er.evaluate_full_sequence(y_test_dir,seq2seq_pred,70)

import mlflow.keras
dir_model = mlflow.keras.load_model("models:/Direct_LSTM_Model_average/1")  

X_train_ref = dir_model.predict(X_train_dir)
y_train_ref = y_train_dir[:, -1].reshape(-1, 1)
X_val_ref = dir_model.predict(X_val_dir)
y_val_ref = y_val_dir[:, -1].reshape(-1, 1)
X_test_ref = dir_model.predict(X_test_dir)
y_test_ref = y_test_dir[:, -1].reshape(-1, 1)
import importlib
from autoregression.models import RefineLSTM
importlib.reload(RefineLSTM)
from autoregression.models.RefineLSTM import build_refinement_model_lstm_avg
ref_model=build_refinement_model_lstm_avg()
ref_model.summary()
ref_history=train_and_save_model(ref_model, X_train=X_train_ref, y_train=y_train_ref, X_val=X_val_ref, y_val=y_val_ref,name='simple_ref.keras',epochs=50)
refinement_model, refinement_history = train_and_save_model(
    model=ref_model,
    X_train=[X_train_dir, X_train_ref], 
    y_train=y_train_ref,
    X_val=[X_val_dir, X_val_ref], 
    y_val=y_val_ref,
    name='refinement_lstm_model.keras',
    epochs=50,
    batch_size=32
)
plot_training_history(refinement_history)
ref_pred=ref_model.predict([X_test_dir,X_test_ref]).flatten()
er.evaluate_n_values(y_test_ref.flatten(),ref_pred,n=20)
from autoregression.utils_ar import run_direct_refiner_experiment
from autoregression.models import RefineLSTM
from autoregression.models.RefineLSTM import build_refinement_cnn_model_simple,build_refinement_lstm_model_simple
importlib.reload(RefineLSTM)
run_direct_refiner_experiment(build_refinement_cnn_model_simple,
                              dir_model,
                              'avg_cnn_',
                              'avg_cnn_',
                              X_train_dir=X_train_dir,
                              X_test_dir=X_test_dir,
                              X_val_dir=X_val_dir,
                              y_test_dir=y_test_dir,
                              y_train_dir=y_train_dir,
                              y_val_dir=y_val_dir,
                              output_steps=20,
                              units=32)


for n_past_values in [100]:
    n_future_values=20
    input_shape=(n_past_values,1)

    X_train_dir, y_train_dir,X_val_dir,y_val_dir,X_test_dir,y_test_dir = prepare_direct_lstm_data(data,n_past_values,n_future_values)
    dir_model = build_direct_lstm_model_simple(input_shape=input_shape, output_steps=n_future_values,lstm_units=100)
    dir_model.summary()
    dir_history = train_and_save_model(dir_model,X_train=X_train_dir, y_train=y_train_dir, X_val=X_val_dir, y_val=y_val_dir,name=f'{n_past_values}_simple_dir,batch.keras',epochs=100,batch_size=64)
    print(f"n_past_vals={n_past_values}")
    plot_training_history(dir_history[1])
    dir_pred=dir_model.predict(X_test_dir)
    er.evaluate_n_values(y_test_dir[:,-1],dir_pred[:,-1],n=20)
    er.evaluate_full_sequence(y_test_dir,dir_pred,2)






model_uri = f"models:/Direct_LSTM_Model_simple/1"  # version 1
# or load latest in stage (e.g., "Production", "Staging")
# model_uri = "models:/Direct_LSTM_Model_simple/Production"

loaded_model = mlflow.keras.load_model(model_uri)
loaded_pred=loaded_model.predict(X_test_dir)
results_df = er.evaluate_n_values(y_test_dir[:,-1], loaded_pred[:,-1], n=20)

       
run_direct_lstm_experiment(
    build_direct_lstm_model_simple,
    'simple_gaus',
    'direct_simple_test',
    X_train_dir,
    y_train_dir,
    X_val_dir,
    y_val_dir,
    X_test_dir,
    y_test_dir,
    20,epochs=100)




run_recursive_lstm_experiment(
    build_recursive_lstm_simple,
    'simple_gaus',
    'recursive_simple_test',
    X_train_rec,
    y_train_rec,
    X_val_rec,
    y_val_rec,
    X_test_rec,
    y_test_dir,
    n_predicted_rows=5000,
    output_steps=20,
    epochs=100)


def run_direct_refiner_lstm_experiment(
    model_builder_func,
    direct_model,
    model_structure_name: str,
    #experiment_name: str,
    run_name: str,
    X_train_dir, y_train_dir, X_val_dir, y_val_dir, X_test_dir, y_test_dir,
    output_steps: int,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    plot_history_func=True,
    log_additional_metrics=True,
    extra_params: dict = None
):
    """
    Run an LSTM training and evaluation experiment with MLflow tracking.

    Parameters:
    - model_builder_func: function to build the model
    - direct_model: trained direct model 
    - model_structure_name: registered model name
    - X_train, y_train, X_val, y_val, X_test, y_test: datasets  for direct model
    - output_steps: number of output steps
    - lstm_units: number of LSTM units
    - batch_size, epochs: training settings
    - plot_history_func : Plot loss history of training 
    - log_additional_metrics: whether to log MSE/MAE/R2 etc.
    - extra_params: dict of additional hyperparameters to log
    """
    X_train = direct_model.predict(X_train_dir)
    y_train = y_train_dir[:, -1]
    X_val = direct_model.predict(X_val_dir)
    y_val = y_val_dir[:, -1]
    X_test = direct_model.predict(X_test_dir)
    y_test = y_test_dir[:, -1]
    input_shape=X_train.shape[1:]
    print(input_shape)
    experiment_name = "Direct_Refiner_Experiments"
    run_name = f"Direct_Refiner{model_structure_name}_{lstm_units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Direct_Refiner_Model_{model_structure_name}"
    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(input_shape=input_shape, output_steps=output_steps, lstm_units=lstm_units)
        model.summary()

        history = train_and_save_model(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            name=f"{registered_model_name}.keras",
            epochs=epochs,
            batch_size=batch_size
        )
        if plot_history_func:
            plot_training_history(history[1])
        # Predictions
        preds = model.predict(X_test)

        # Evaluation
        # adjust import if needed
        results_df = er.evaluate_n_values(y_test, preds, n=output_steps)

        # Signature and input example
        input_example = X_test[0:1]
        pred_example = model.predict(input_example)
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature
        )

        # Log training metrics
        mlflow.log_metric("val_loss", history[1].history['val_loss'][-1])
        if "val_mae" in history[1].history:
            mlflow.log_metric("val_mae", history[1].history["val_mae"][-1])
        # Log evaluation metrics
        if log_additional_metrics:
            for index, row in results_df.iterrows():
                mlflow.log_metric(row['Metric'], row['Value'])

        # Log hyperparameters
        mlflow.log_params({
            "structure": model_structure_name,
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            **(extra_params or {})
        })
        mlflow.end_run()    
















from autoregression.utils_ar import run_direct_refiner_experiment
from autoregression.models.RefineLSTM import build_refinement_cnn_model_simple,build_refinement_lstm_model_simple
importlib.reload(utils_ar)






































from datetime import datetime
import mlflow
from mlflow.models import infer_signature

def run_seq2seq_scheduled_sampling_experiment(
    model_builder_func,
    model_structure_name: str,
    run_name: str,
    X_train_enc, X_train_dec, y_train,
    X_val_enc, X_val_dec, y_val,
    X_test_enc, y_test,
    output_steps: int,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    sampling_decay: float = 0.95,
    min_sampling_rate: float = 0.1,
    log_additional_metrics=True,
    extra_params: dict = None,
    ML_FLOW_DIR="mlruns"
):
    experiment_name = "Seq2Seq_ScheduledSampling_Experiments"
    run_name = f"Seq2seqScheduled_{model_structure_name}_{lstm_units}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    registered_model_name = f"Seq2SeqScheduled_LSTM_Model_{model_structure_name}"

    mlflow.set_tracking_uri(ML_FLOW_DIR)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        model = model_builder_func(
            input_timesteps=X_train_enc.shape[1],
            output_timesteps=output_steps,
            n_features=X_train_enc.shape[2],
            lstm_units=lstm_units,
            sampling_decay=sampling_decay,
            min_sampling_rate=min_sampling_rate
        )
        model.summary()

        # Train using custom function
        history = train_scheduled_seq2seq_model(
            model,
            X_encoder=X_train_enc,
            X_decoder=X_train_dec,
            y_target=y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Log last loss
        mlflow.log_metric("final_train_loss", history[-1])

        # Predict on test
        y_pred = model(X_test_enc, training=False).numpy()

        # Evaluation
        results_df = er.evaluate_n_values(y_test[:,-1], y_pred[:,-1], n=output_steps)
        er.evaluate_full_sequence(y_test, y_pred, 0)

        # Model logging
        input_example = X_test_enc[:1]
        pred_example = model(input_example, training=False).numpy()
        signature = infer_signature(input_example, pred_example)

        mlflow.keras.log_model(
            model,
            artifact_path=registered_model_name,
            registered_model_name=registered_model_name,
            signature=signature
        )

        # Log metrics
        for index, row in results_df.iterrows():
            mlflow.log_metric(row['Metric'], row['Value'])

        # Log params
        mlflow.log_params({
            "structure": model_structure_name,
            "lstm_units": lstm_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_steps": output_steps,
            "sampling_decay": sampling_decay,
            "min_sampling_rate": min_sampling_rate,
            **(extra_params or {})
        })

        mlflow.end_run()
        
        
        
        
import tensorflow as tf

class ScheduledSamplingSeq2Seq(tf.keras.Model):
    def __init__(self, input_timesteps, output_timesteps, n_features, lstm_units=64, sampling_decay=0.95, min_sampling_rate=0.1):
        super(ScheduledSamplingSeq2Seq, self).__init__()
        self.encoder_lstm = tf.keras.layers.LSTM(lstm_units, return_state=True)
        self.decoder_lstm_cell = tf.keras.layers.LSTMCell(lstm_units)
        self.output_layer = tf.keras.layers.Dense(n_features)
        self.output_timesteps = output_timesteps
        self.sampling_rate = 1.0
        self.sampling_decay = sampling_decay
        self.min_sampling_rate = min_sampling_rate
        self.n_features = n_features

    def call(self, encoder_inputs, decoder_inputs=None, training=True):
        batch_size = tf.shape(encoder_inputs)[0]
        _, state_h, state_c = self.encoder_lstm(encoder_inputs)
        decoder_state = [state_h, state_c]

        decoder_input = tf.expand_dims(tf.zeros((batch_size, self.n_features)), axis=1)

        all_outputs = []
        for t in range(self.output_timesteps):
            output, decoder_state = self.decoder_lstm_cell(tf.squeeze(decoder_input, axis=1), states=decoder_state)
            output = self.output_layer(output)
            all_outputs.append(output)

            if training and decoder_inputs is not None:
                use_teacher = tf.random.uniform([]) < self.sampling_rate
                teacher_input = decoder_inputs[:, t:t+1, :]
                output_input = tf.expand_dims(output, axis=1)
                decoder_input = tf.where(
                    tf.broadcast_to(use_teacher, tf.shape(teacher_input)),
                    teacher_input,
                    output_input
                )
            else:
                decoder_input = tf.expand_dims(output, axis=1)

        return tf.stack(all_outputs, axis=1)

    def update_sampling_rate(self):
        self.sampling_rate = max(self.sampling_rate * self.sampling_decay, self.min_sampling_rate)
       
        
        
        
        
def train_scheduled_seq2seq_model(model, X_encoder, X_decoder, y_target,
                                  epochs=20, batch_size=32,
                                  loss_fn=None, optimizer=None,
                                  verbose=True):
    if loss_fn is None:
        loss_fn = tf.keras.losses.MeanSquaredError()
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_encoder, X_decoder, y_target))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, (enc_batch, dec_batch, target_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(enc_batch, dec_batch, training=True)
                loss = loss_fn(target_batch, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()

        model.update_sampling_rate()
        avg_loss = epoch_loss / (step + 1)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Sampling Rate: {model.sampling_rate:.2f}")

    return loss_history