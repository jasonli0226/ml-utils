import tensorflow as tf
from ml_utils.estimator import net_flops, time_per_layer

model = tf.keras.applications.VGG16(
    weights=None, include_top=True, pooling=None, input_shape=(224, 224, 3)
)

net_flops(model, table=True)
time_per_layer(model, visualize=True)
