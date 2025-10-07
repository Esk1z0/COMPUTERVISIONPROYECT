# Model 1

## Architecture

```python
model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))
model.add(LayerNormalization())
# Dense block 1
model.add(Dense(1024, activation='gelu', kernel_regularizer=l2(1e-5))) # Dense layers were originally x2 neurons
model.add(LayerNormalization())
model.add(Dropout(0.3))
# Dense block 2
model.add(Dense(512, activation='gelu', kernel_regularizer=l2(1e-5)))
model.add(LayerNormalization())
model.add(Dropout(0.3))
# Dense block 3
model.add(Dense(256, activation='gelu', kernel_regularizer=l2(1e-5)))
model.add(LayerNormalization())
model.add(Dropout(0.3))
# Dense block 4
model.add(Dense(128, activation='gelu', kernel_regularizer=l2(1e-5)))
model.add(LayerNormalization())
model.add(Dropout(0.3))
# Output layer
model.add(Dense(len(categories), activation='softmax'))

opt = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Hyperparameters

```python
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
REGULARIZER_PENALTY = 1e-5
EPOCHS = 50
```

## Callbacks

```python
model_checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping('val_loss', patience=10, verbose=1, restore_best_weights=True)
terminate = TerminateOnNaN()
```

## Validation results

Mean Accuracy: 60.320%
Mean Recall: 53.456%
Mean Precision: 56.517%
> Cargo plane: Recall: 83.051% Precision: 90.741% Specificity: 99.725% Dice: 86.726%
> Small car: Recall: 91.793% Precision: 62.656% Specificity: 88.357% Dice: 74.476%
> Bus: Recall: 55.172% Precision: 51.337% Specificity: 94.650% Dice: 53.186%
> Truck: Recall: 15.768% Precision: 33.043% Specificity: 95.288% Dice: 21.348%
> Motorboat: Recall: 36.697% Precision: 48.780% Specificity: 97.622% Dice: 41.885%
> Fishing vessel: Recall: 50.685% Precision: 74.000% Specificity: 99.279% Dice: 60.163%
> Dump truck: Recall: 59.322% Precision: 50.725% Specificity: 96.130% Dice: 54.688%
> Excavator: Recall: 53.086% Precision: 60.563% Specificity: 98.439% Dice: 56.579%
> Building: Recall: 76.860% Precision: 67.883% Specificity: 91.270% Dice: 72.093%
> Helipad: Recall: 0.000% Precision: 0.000% Specificity: 100.000% Dice: 0.000%
> Storage tank: Recall: 50.000% Precision: 72.381% Specificity: 98.317% Dice: 59.144%
> Shipping container: Recall: 60.000% Precision: 53.642% Specificity: 95.977% Dice: 56.643%
> Pylon: Recall: 62.500% Precision: 68.966% Specificity: 99.512% Dice: 65.574%