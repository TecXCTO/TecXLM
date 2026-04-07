import tensorflow as tf
import tf2onnx

# 1. Load your trained Keras model
model = tf.keras.models.load_model('my_model.h5')

# 2. Convert to ONNX format
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),) # Match your input shape
output_path = "model_tf.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print("TensorFlow model converted to model_tf.onnx")
