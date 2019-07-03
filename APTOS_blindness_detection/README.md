# APTOS blindness detection kaggle competition


In this competition, the objective is to correctly classify diabetic retinopathy in images taken using a particular form of photography. There are five ratings that represent the various diagnoses for each image:
 * 0 - No diabetic retinopathy (DR)
 * 1 - Mild DR
 * 2 - Moderate DR
 * 3 - Severe DR
 * 4 - Proliferative DR

Below are some of the approaches we can try to classify these images:
  * Train a convolutional neural network (convnet) from scrath.
  * Train a convnet from scratch but additionally augment the data by applying transformations available in keras.
  * Use a pretrained network for image classification tasks to extract features from the competition data, then feed those through only the classification layer of the convnet.
  * Use a pretrained network but train from end-to-end, which will be much more computationally expensive but allows for data augmentation which may be beneficial because we only have a few thousand images per class.


Example images:
![img1](img/comps-14774-536888-train_images-000c1434d8d7.png)
