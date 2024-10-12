import tensorflow as tf
print(f'TF version: {tf.__version__}')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Extract the GPU indices from the device names
    gpu_indices = [int(gpu.name.split(':')[-1]) for gpu in gpus]
    print("GPU Indices:", gpu_indices)
else:
    print("No GPUs available in your system.")