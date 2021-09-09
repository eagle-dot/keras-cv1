"# Keras-cv1" 

This code is extracted from Keras 

https://keras.io/examples/vision/image_classification_from_scratch/

1) download data 

go to the install directory and run the following command 

C:\<install-dir> 

curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip -q kagglecatsanddogs_3367a.zip

example fro the install-dir
C:\Users\mitchell\PycharmProjects\BT1-ph1\keras>

2) download the source code to your install_dir

3) you might need to upgrade your tensorflow 
pip install tensorflow --upgrade


I am using Pycharm 2020.3 and run the code on  Windows 10 (GPU:Quadro P2200 )

The program takes about 4 hours to complete ( mainly the model training) 

Here is my output


Found 23422 files belonging to 2 classes.
Using 18738 files for training.
2021-09-08 16:03:55.998894: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-09-08 16:03:56.009843: I tensorflow/compiler/xla/service/service.cc:171] XLA service 0x1e0fd4733a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-09-08 16:03:56.010332: I tensorflow/compiler/xla/service/service.cc:179]   StreamExecutor device (0): Host, Default Version
2021-09-08 16:03:56.622459: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3619 MB memory:  -> device: 0, name: Quadro P2200, pci bus id: 0000:65:00.0, compute capability: 6.1
2021-09-08 16:03:56.626613: I tensorflow/compiler/xla/service/service.cc:171] XLA service 0x1e0d8b1c600 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-09-08 16:03:56.626726: I tensorflow/compiler/xla/service/service.cc:179]   StreamExecutor device (0): Quadro P2200, Compute Capability 6.1
Found 23422 files belonging to 2 classes.
Using 4684 files for validation.
2021-09-08 16:03:58.431072: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
display. Cat
display. Cat
display ...Dog
display ...Dog
display. Cat
display. Cat
display ...Dog
display. Cat
display. Cat
Epoch 1/50
2021-09-08 16:04:04.631825: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8101
2021-09-08 16:04:07.108580: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:07.228709: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.34GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:07.314780: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.27GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:07.885542: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.17GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:07.909473: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.27GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:08.032934: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.14GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:08.055952: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.34GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:08.261468: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.17GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-09-08 16:04:08.300348: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.55GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
 66/586 [==>...........................] - ETA: 3:23 - loss: 0.7320 - accuracy: 0.5881Warning: unknown JFIF revision number 0.00
110/586 [====>.........................] - ETA: 3:05 - loss: 0.7118 - accuracy: 0.5989Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
261/586 [============>.................] - ETA: 2:06 - loss: 0.6592 - accuracy: 0.6371Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
291/586 [=============>................] - ETA: 1:55 - loss: 0.6508 - accuracy: 0.6442Corrupt JPEG data: 228 extraneous bytes before marker 0xd9
430/586 [=====================>........] - ETA: 1:01 - loss: 0.6171 - accuracy: 0.6703Corrupt JPEG data: 239 extraneous bytes before marker 0xd9
510/586 [=========================>....] - ETA: 29s - loss: 0.6056 - accuracy: 0.6790Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
585/586 [============================>.] - ETA: 0s - loss: 0.5925 - accuracy: 0.68922021-09-08 16:07:57.890681: W tensorflow/core/common_runtime/bfc_allocator.cc:272] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.33GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
586/586 [==============================] - ETA: 0s - loss: 0.5924 - accuracy: 0.6892Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
586/586 [==============================] - 250s 415ms/step - loss: 0.5924 - accuracy: 0.6892 - val_loss: 0.5209 - val_accuracy: 0.7477
C:\Users\mitchell\PycharmProjects\BT1-ph1\venv\lib\site-packages\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
  warnings.warn('Custom mask layers require a config and must override '

....




586/586 [==============================] - 243s 414ms/step - loss: 0.0543 - accuracy: 0.9790 - val_loss: 0.1128 - val_accuracy: 0.9633
Epoch 50/50
 66/586 [==>...........................] - ETA: 3:24 - loss: 0.0423 - accuracy: 0.9834Warning: unknown JFIF revision number 0.00
110/586 [====>.........................] - ETA: 3:07 - loss: 0.0455 - accuracy: 0.9827Corrupt JPEG data: 252 extraneous bytes before marker 0xd9
261/586 [============>.................] - ETA: 2:08 - loss: 0.0484 - accuracy: 0.9823Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
291/586 [=============>................] - ETA: 1:56 - loss: 0.0479 - accuracy: 0.9822Corrupt JPEG data: 228 extraneous bytes before marker 0xd9
430/586 [=====================>........] - ETA: 1:01 - loss: 0.0470 - accuracy: 0.9822Corrupt JPEG data: 239 extraneous bytes before marker 0xd9
510/586 [=========================>....] - ETA: 29s - loss: 0.0495 - accuracy: 0.9812Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
586/586 [==============================] - ETA: 0s - loss: 0.0496 - accuracy: 0.9812Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Corrupt JPEG data: 2226 extraneous bytes before marker 0xd9
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
586/586 [==============================] - 243s 414ms/step - loss: 0.0496 - accuracy: 0.9812 - val_loss: 0.1108 - val_accuracy: 0.9620
This image is 75.73 percent cat and 24.27 percent dog.
