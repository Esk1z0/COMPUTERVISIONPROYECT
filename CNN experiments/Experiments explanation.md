# CNN Challenge

Partimos de lo que habíamos logrado en el anterior challenge

- To add in some point
    
    **Regularization**
    
    ![image.png](attachment:7e531918-f0d7-49b7-96bc-ff2171ed8159:image.png)
    

# Basic CNN

We are going to create a basic CNN with the stratified dataset

This CNN won’t use a Classification layer

We will start with 20 epochs training

The architecture we are using is 3 convolutional layers with relu activation and padding=same

After each convolution we have a pooling 2x2 and we end with a flatten layer followed by the output layer. We have no classification layer in between

## Configuración

### Arquitectura modelo

```python
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool3'))

model.add(Flatten(name='Flatten'))

model.add(Dense(len(categories), activation='softmax'))

model.summary()
```

```python
opt = Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Hiperparámetros

```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = None
DROPOUT_RATE = None
REGULARIZER_PENALTY = None
EPOCHS = 20
```

### Callbacks

```python
# Ahora sin Early Stopping
model_checkpoint = ModelCheckpoint(BEST_WEIGHTS_FILENAME, monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping('val_loss', patience=EPOCHS, verbose=1, restore_best_weights=True)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

### Datos

```python
# Train/Validation stratified split of 10% for validation
# Train shuffled
train_dataset = create_tfrecord_dataset([TRAIN_SET_PATH], BATCH_SIZE, do_shuffle=True)
valid_dataset = create_tfrecord_dataset([VALIDATION_SET_PATH], BATCH_SIZE)
```

## Resultados

### Matriz de confusión

![image.png](attachment:6d954e5d-0d14-40ef-abf6-5f03c772e03a:image.png)

### Métricas

| **Experiment** | **CNN/1_basic_cnn** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Category** | **Global** | **Building** | **Small car** | **Truck** | **Bus** | **Shipping container** | **Storage tank** | **Dump truck** | **Motorboat** | **Excavator** | **Fishing vessel** | **Cargo plane** | **Pylon** | **Helipad** |
| Size | 1875.000000 | 359.000000 | 332.000000 | 221.000000 | 177.000000 | 152.000000 | 147.000000 | 124.000000 | 107.000000 | 79.000000 | 71.000000 | 64.000000 | 31.000000 | 11.000000 |
| Recall | 0.503569 | 0.721448 | 0.798193 | 0.149321 | 0.480226 | 0.618421 | 0.231293 | 0.161290 | 0.457944 | 0.607595 | 0.098592 | 0.843750 | 0.612903 | 0.272727 |
| Precision | 0.503569 | 0.674479 | 0.629454 | 0.227586 | 0.277778 | 0.467662 | 0.641509 | 0.384615 | 0.494949 | 0.521739 | 0.333333 | 0.870968 | 0.593750 | 0.428571 |
| Specificity | NaN | 0.917546 | 0.898898 | 0.932285 | 0.869847 | 0.937899 | 0.989005 | 0.981725 | 0.971719 | 0.975501 | 0.992239 | 0.995583 | 0.992950 | 0.997854 |
| F1 Score | NaN | 0.697174 | 0.703851 | 0.180328 | 0.351967 | 0.532578 | 0.340000 | 0.227273 | 0.475728 | 0.561404 | 0.152174 | 0.857143 | 0.603175 | 0.333333 |
| Accuracy | 0.465669 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

### Entrenamiento

![image.png](attachment:02843a81-a7e8-4e6e-a1d7-8cf05312518f:image.png)

## Conclusiones

Obtenemos un modelo que enseguida llega al overfitting. Para remediarlo aplicaremos regularización mediante data augmentation.

# Data agumentation

We will add data augmentation as the model optimized so hard

## Configuración

### Arquitectura modelo

```python
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool3'))

model.add(Flatten(name='Flatten'))

model.add(Dense(len(categories), activation='softmax'))

model.summary()
```

```python
opt = Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Hiperparámetros

```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = None
DROPOUT_RATE = None
REGULARIZER_PENALTY = None
DATA_AUGMENTATAION_RATE = 0.2
EPOCHS = 20
```

### Callbacks

```python
# Ahora sin Early Stopping
model_checkpoint = ModelCheckpoint(BEST_WEIGHTS_FILENAME, monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping('val_loss', patience=EPOCHS, verbose=1, restore_best_weights=True)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

### Datos

```python
# Train/Validation stratified split of 10% for validation
# Train shuffled
train_dataset = create_tfrecord_dataset([TRAIN_SET_PATH], BATCH_SIZE, do_shuffle=True)
valid_dataset = create_tfrecord_dataset([VALIDATION_SET_PATH], BATCH_SIZE)
```

```python
# Data augmentation
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(DATA_AUGMENTATAION_RATE),
    RandomZoom(DATA_AUGMENTATAION_RATE),
])
...
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))
model.add(data_augmentation)
...
```

## Resultados

### Matriz de confusión

![image.png](attachment:2b6a28c6-8b8b-4995-9b12-deaf57dad0f9:image.png)

### Métricas

| **Experiment** | **CNN/2_data_augmentation** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Category** | **Global** | **Building** | **Small car** | **Truck** | **Bus** | **Shipping container** | **Storage tank** | **Dump truck** | **Motorboat** | **Excavator** | **Fishing vessel** | **Cargo plane** | **Pylon** | **Helipad** |
| Size | 1875.000000 | 359.000000 | 332.000000 | 221.000000 | 177.000000 | 152.000000 | 147.000000 | 124.000000 | 107.000000 | 79.000000 | 71.000000 | 64.000000 | 31.000000 | 11.000000 |
| Recall | 0.535800 | 0.732591 | 0.777108 | 0.203620 | 0.355932 | 0.611842 | 0.312925 | 0.096774 | 0.383178 | 0.443038 | 0.169014 | 0.812500 | 0.419355 | 0.363636 |
| Precision | 0.535800 | 0.589686 | 0.578475 | 0.254237 | 0.386503 | 0.353612 | 0.407080 | 0.387097 | 0.611940 | 0.555556 | 0.521739 | 0.928571 | 0.590909 | 0.800000 |
| Specificity | NaN | 0.879288 | 0.878159 | 0.920193 | 0.941107 | 0.901335 | 0.961227 | 0.989149 | 0.985294 | 0.984410 | 0.993902 | 0.997791 | 0.995119 | 0.999464 |
| F1 Score | NaN | 0.653416 | 0.663239 | 0.226131 | 0.370588 | 0.448193 | 0.353846 | 0.154839 | 0.471264 | 0.492958 | 0.255319 | 0.866667 | 0.490566 | 0.500000 |
| Accuracy | 0.437039 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

### Entrenamiento

![image.png](attachment:fdaeb8cc-f32a-437f-a9b4-11d87fae29f5:image.png)

## Conclusiones

We reduced the overfitting significantly

Now we can try to add a classification layer after the convolutional part

This way we should expect to obtain better results than before

We will start by an easy dense layer of 512 units with relu activation function

# Classification layer

We will add a classification layer to increase the performance of the NN

## Configuración

### Arquitectura modelo

```python
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool3'))

model.add(Flatten(name='Flatten'))

model.add(Dense(512, activation='relu'))

model.add(Dense(len(categories), activation='softmax'))

model.summary()
```

```python
opt = Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Hiperparámetros

```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = None
DROPOUT_RATE = None
REGULARIZER_PENALTY = None
DATA_AUGMENTATAION_RATE = 0.2
EPOCHS = 20
```

### Callbacks

```python
# Ahora sin Early Stopping
model_checkpoint = ModelCheckpoint(BEST_WEIGHTS_FILENAME, monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping('val_loss', patience=EPOCHS, verbose=1, restore_best_weights=True)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

### Datos

```python
# Train/Validation stratified split of 10% for validation
# Train shuffled
train_dataset = create_tfrecord_dataset([TRAIN_SET_PATH], BATCH_SIZE, do_shuffle=True)
valid_dataset = create_tfrecord_dataset([VALIDATION_SET_PATH], BATCH_SIZE)
```

```python
# Data augmentation
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(DATA_AUGMENTATAION_RATE),
    RandomZoom(DATA_AUGMENTATAION_RATE),
])
...
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))
model.add(data_augmentation)
...
```

## Resultados

### Matriz de confusión

![image.png](attachment:fb04de98-dfe0-4f44-ac2a-9d4cd575d73f:image.png)

### Métricas

| **Experiment** | **CNN/3_classification_layer** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Category** | **Global** | **Building** | **Small car** | **Truck** | **Bus** | **Shipping container** | **Storage tank** | **Dump truck** | **Motorboat** | **Excavator** | **Fishing vessel** | **Cargo plane** | **Pylon** | **Helipad** |
| Size | 1875.000000 | 359.000000 | 332.000000 | 221.000000 | 177.000000 | 152.000000 | 147.000000 | 124.000000 | 107.000000 | 79.000000 | 71.000000 | 64.000000 | 31.000000 | 11.000000 |
| Recall | 0.521513 | 0.813370 | 0.828313 | 0.063348 | 0.288136 | 0.447368 | 0.244898 | 0.233871 | 0.523364 | 0.531646 | 0.450704 | 0.734375 | 0.354839 | 0.363636 |
| Precision | 0.521513 | 0.562620 | 0.516917 | 0.274510 | 0.401575 | 0.352332 | 0.404494 | 0.446154 | 0.533333 | 0.583333 | 0.727273 | 0.921569 | 0.611111 | 0.444444 |
| Specificity | NaN | 0.850264 | 0.833441 | 0.977630 | 0.955241 | 0.927452 | 0.969329 | 0.979440 | 0.972285 | 0.983296 | 0.993348 | 0.997791 | 0.996204 | 0.997318 |
| F1 Score | NaN | 0.665148 | 0.636574 | 0.102941 | 0.335526 | 0.394203 | 0.305085 | 0.306878 | 0.528302 | 0.556291 | 0.556522 | 0.817391 | 0.448980 | 0.400000 |
| Accuracy | 0.452144 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

### Entrenamiento

![image.png](attachment:04554e5c-9d87-40fa-a9f4-947de759eb4a:image.png)

## Conclusiones

Looks like we are facing the same problem we had when first time introducing hidden layers. These layers started to optimize over the predominant classes (i.e. Building and Small car) we can see it on the Precision metric of these classes droping down from 0.67 and 0.62 to 0.56 and 0.51 respectively.

To deal with this we will try to add a rescaling to the input going from the range [0.255] to [0,1].

We can also see that on those first epochs the model significantly droped its performance w.rt. to the previous experiments.

This scaling on the input could help on improving this first phaseson the model

# Reescaling

As we had some issues of focusing so hard on predominant classes and also on learning on those first epochs we will try to reescale the input to the [0,1] range

## Configuración

### Arquitectura modelo

```python
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))
model.add(Rescaling(1./255))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), name='Pool3'))

model.add(Flatten(name='Flatten'))

model.add(Dense(512, activation='relu'))

model.add(Dense(len(categories), activation='softmax'))

model.summary()
```

```python
opt = Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Hiperparámetros

```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = None
DROPOUT_RATE = None
REGULARIZER_PENALTY = None
DATA_AUGMENTATAION_RATE = 0.2
EPOCHS = 20
```

### Callbacks

```python
# Ahora sin Early Stopping
model_checkpoint = ModelCheckpoint(BEST_WEIGHTS_FILENAME, monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping('val_loss', patience=EPOCHS, verbose=1, restore_best_weights=True)
terminate = TerminateOnNaN()
callbacks = [model_checkpoint, reduce_lr, early_stop, terminate]
```

### Datos

```python
# Train/Validation stratified split of 10% for validation
# Train shuffled
train_dataset = create_tfrecord_dataset([TRAIN_SET_PATH], BATCH_SIZE, do_shuffle=True)
valid_dataset = create_tfrecord_dataset([VALIDATION_SET_PATH], BATCH_SIZE)
```

```python
# Data augmentation
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(DATA_AUGMENTATAION_RATE),
    RandomZoom(DATA_AUGMENTATAION_RATE),
])
...
model = Sequential(name=EXPERIMENT_NAME)
model.add(Input(shape=(224, 224, 3)))
model.add(data_augmentation)
...
```

## Resultados

### Matriz de confusión

![image.png](attachment:b0b023ce-6b5c-490d-92a4-1a99ce306034:image.png)

### Métricas

| **Experiment** | **CNN/4_reescaling** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Category** | **Global** | **Building** | **Small car** | **Truck** | **Bus** | **Shipping container** | **Storage tank** | **Dump truck** | **Motorboat** | **Excavator** | **Fishing vessel** | **Cargo plane** | **Pylon** | **Helipad** |
| Size | 1875.000000 | 359.000000 | 332.000000 | 221.000000 | 177.000000 | 152.000000 | 147.000000 | 124.000000 | 107.000000 | 79.000000 | 71.000000 | 64.000000 | 31.000000 | 11.000000 |
| Recall | 0.659657 | 0.846797 | 0.888554 | 0.239819 | 0.384181 | 0.519737 | 0.544218 | 0.467742 | 0.691589 | 0.772152 | 0.718310 | 0.890625 | 0.709677 | 0.545455 |
| Precision | 0.659657 | 0.739659 | 0.631692 | 0.339744 | 0.544000 | 0.738318 | 0.629921 | 0.483333 | 0.718447 | 0.622449 | 0.850000 | 0.919355 | 0.758621 | 0.600000 |
| Specificity | NaN | 0.929420 | 0.888529 | 0.937727 | 0.966431 | 0.983749 | 0.972801 | 0.964592 | 0.983597 | 0.979399 | 0.995011 | 0.997239 | 0.996204 | 0.997854 |
| F1 Score | NaN | 0.789610 | 0.738423 | 0.281167 | 0.450331 | 0.610039 | 0.583942 | 0.475410 | 0.704762 | 0.689266 | 0.778626 | 0.904762 | 0.733333 | 0.571429 |
| Accuracy | 0.632220 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

### Entrenamiento

![image.png](attachment:c35da301-c9cf-484b-a2d4-6cfa5361d5d5:image.png)

## Conclusiones

We have seen this big improvement on the models performance at the beggining

This input’s reescale has benefit it so much

We have seen that the Precision metrics on the most populated classes has risen again even over to the results obtained at the beggining. Now at 0.73 and 0.63, both over the initial 0.67 and 0.62.

Now we can see that we start to still have some overfitting in the latter epochs. To address this we can try to add some dropout on the connection between the Convolution and Dense layers.