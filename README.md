# Convolutional-Neural-Networks-using-Tensorflow 
### Week 1
<pre>
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
</pre>
<br>
<pre>
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_generator.flow_from_directory(
                    train_dir,
                    target_size=(150,150),
                    batch_size=20,
                    class_mode='binary')
</pre>
<br>
<pre>
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_generator.flow_from_directory(
                    validation_dir,
                    target_size=(150,150),
                    batch_size=20,
                    class_mode='binary')
</pre>
<br>
<pre>
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
</pre>
<br>
<pre>
from tensorflow.keras.optimizers import RMSProp
model.compile(loss='binary_crossentropy',
      optimizer=RMSProp(lr=0.001),
      metrics=['acc'])
</pre>
<br>
<pre>
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=15,
        validation_data = validation_generator,
        validation_steps=50,
        verbose=2)
</pre>
<br><br><br><br><br><br><br><br>
<pre>
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
</pre>
<pre>

</pre>
