# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import mlflow 
import mlflow.tensorflow
import mlflow.keras
import tensorflow as tf 
import keras
from tensorflow.keras.datasets import cifar10 
import matplotlib.pyplot as plt


# %%
# Loading the dataset
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# %%
x_train.shape
x_test.shape
y_train.shape
y_test.shape


# %%
# Normalizing the images
x_train = x_train / 255.0
x_test = x_test / 255.0


# %%
plt.imshow(x_train[1])
print(classes[y_train[1,0]])


# %%
mlflow.keras.autolog()


# %%
# Builiding a CNN model 

def run_model(params):

    with mlflow.start_run(run_name ="Monitoring CNN model training") as run:  
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
        cnn_model_2 .compile(optimizer ="Adam", loss="sparse_categorical_crossentropy", metrics =['accuracy'])
        # Fit the model
        cnn_model_2.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = params['epochs'])
        return (run.info.experiment_id, run.info.run_id)
        


# %%
for epochs, conv_size in [[3,3],[2,2]]:
    params = {'epochs': epochs, 'conv_size': conv_size}
    run_model(params)


# %%



