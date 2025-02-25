import tensorflow as tf

model = tf.keras.models.load_model("Leaf_Model")  # Load your existing model
model.save("Leaf_Model.keras")  # Save it in the new format
