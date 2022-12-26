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
### Week 3
<pre>
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
</pre>
<br>
<pre>
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weight_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights = None)

pre_trained_model.load_weights(local_weights_file)
</pre>
<br>
<pre>
for layer in pre_trained_model.layers:
  layer.trainable=False
</pre>
<br>
<pre>
pre_trained_model.summary()
</pre>
<br>
<pre>
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
</pre>
<br>
<pre>
from tensorflow.keras.optimizers import RMSProp

x= layers.Flatten()(last_output)
x= layers.Dense(1024, activation='relu')(x)
x= layers.Dense(1,activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSProp(learning_rate=0.0001), loss='binary_crossentropy',metrics=['acc'])
</pre>
<br>
<pre>
train_datagen = ImageDataGenerator(rescale=1./255., rotation_range = 40, width_shift_range=0.2,
                                    height_shift_range = 0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
</pre>
<br>
<pre>
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size = (150,150))
</pre>
<br>
<pre>
history = model.fit(train_generator,
            validation_data=validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose=2)
</pre>
