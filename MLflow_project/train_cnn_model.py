import mlflow 
import mlflow.tensorflow
import mlflow.keras
import tensorflow as tf 
import keras
from tensorflow.keras.datasets import cifar10 
import matplotlib.pyplot as plt
import sys


# Loading the dataset
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing the images
x_train = x_train / 255.0
x_test = x_test / 255.0



def mlflow_run(params, run_name = "Tracking Experiment: TensorFlow - CNN"):

    with mlflow.start_run(run_name = run_name) as run:  

        # Builiding a CNN model 
        cnn_model_2 = keras.models.Sequential([
                keras.layers.Conv2D(32, kernel_size = params['conv_size'], activation='relu', input_shape=(32,32,3)),
                keras.layers.MaxPool2D(pool_size=(2, 2)),

                keras.layers.Conv2D(filters=64,kernel_size = params['conv_size'],padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2, 2)),

                keras.layers.Conv2D(128, kernel_size = params['conv_size'], activation='relu'),
                keras.layers.MaxPool2D(pool_size=(2, 2)),

                keras.layers.Flatten(),

                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(10, activation='softmax')    
                ])

        # Complie the model
        cnn_model_2 .compile(optimizer ="Adam", loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
        # Fit the model
        model = cnn_model_2.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = params['epochs'])
        model_loss, model_accuracy = cnn_model_2.evaluate(x_test, y_test)
        print(f' Test loss is {model_loss}, Test accuracy is {model_accuracy}')
       

    
        mlflow.log_param("Epochs", params['epochs'])
        mlflow.log_param("conv_size", params['conv_size'])

        mlflow.log_metric("accuracy", model.history['accuracy'][0])
        mlflow.log_metric("loss",model.history['loss'][0] )
        mlflow.log_metric("model_loss", model_loss)
        mlflow.log_metric("model_accuracy", model_accuracy)

        mlflow.keras.log_model(cnn_model_2, "model")

        return (run.info.experiment_id, run.info.run_uuid)

if __name__ == '__main__':
   
   conv_size = tuple(sys.argv[1]) if len(sys.argv) > 1 else (2,2)
   epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
   params = {'epochs': epochs,
            'conv_size': conv_size}
   (exp_id, run_id) = mlflow_run(params)

   print(f"Experiment id={exp_id} and run id = {run_id}")