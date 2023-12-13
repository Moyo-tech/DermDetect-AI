# DermDetect-AI
## Project Overview
### Introduction and Background
Skin cancer is a prevalent health concern globally, with increasing incidences requiring advanced diagnostic tools. Early detection of skin cancer is crucial for effective treatment and improved patient outcomes. This project leverages deep learning techniques to develop an efficient skin cancer classification model, capable of accurately identifying different skin cancer types based on medical images.

### The Problem
<hr>
A dermatologist's first step in determining whether a skin lesion is malignant or benign is to perform a skin biopsy. The dermatologist performs a skin biopsy by removing a portion of the skin lesion and examining it under a microscope. Starting with a dermatologist appointment and ending with a biopsy result, the present process takes nearly a week or more. This research intends to reduce the present gap to a few days by employing Computer-Aided Diagnosis (CAD) to provide the prediction model. The method use a Convolutional Neural Network (CNN) to classify nine forms of skin cancer from photos of outlier lesions. This closing of a gap has the potential to benefit millions of individuals.

### Motivation and Significance
<hr>
The motivation behind this project stems from the potential to revolutionize skin cancer diagnosis. By implementing deep learning algorithms, we aim to create a robust model capable of classifying skin cancer types accurately. Such a model could significantly improve diagnostic efficiency, reduce human error, and ultimately contribute to better patient outcomes.We hope to close the gap between diagnosis and therapy with the help of this research. The successful completion of the research with greater precision on the dataset could benefit the dermatological clinic's work. The model's improved accuracy and efficiency can help detect skin cancer lesion types in its early stages and prevent unnecessary biopsies.

### The Solution
<hr>
Our solution involves the development of a skin cancer classification model based on the DenseNet121 architecture, a deep convolutional neural network known for its effectiveness in image classification tasks. The model is trained on a diverse dataset of skin lesion images, aiming to achieve high accuracy and generalizability. The classification model is trained to distinguish between these different skin cancer lesion types. Each type represents a unique category of skin abnormalities, allowing for precise and targeted diagnosis.

## The Datasets Used
This project used two primary datasets to train and evaluate the skin cancer classification model. These datasets are the Skin Cancer MNIST: HAM10000 from Kaggle and the training set for the ISIC 2018 challenge (Task 3).


**1. Skin Cancer MNIST: HAM10000** :

The [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) obtained from **Kaggle** was created to address the challenges associated with training neural networks for automated diagnosis of pigmented skin lesions. It includes diverse dermatoscopic images acquired from different populations using various modalities. The dataset comprises 10,015 dermatoscopic images, encompassing a range of important diagnostic categories for pigmented lesions.

Ground truth information for the lesions is provided, with over 50% confirmed through histopathology (histo), and the rest determined by follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset also includes lesions with multiple images, allowing tracking through the lesion_id-column within the HAM10000_metadata file.

**3. ISIC 2018 Challenge Training Set (Task 3)**:

The second dataset used in this project is the [training set for the ISIC 2018 challenge](https://challenge.isic-archive.com/data/#2018), precisely Task 3. This dataset complements the HAM10000 dataset, contributing additional images for training the skin cancer classification model.

Refer to the provided [documentation](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and [original sources](https://challenge.isic-archive.com/landing/2018/) for further details about each dataset and their respective licenses.

These two datasets were combined and processed to form a comprehensive and diverse dataset for training and evaluating the skin cancer classification model. The merging of these datasets enhances the model's ability to generalize and make accurate predictions across various skin lesion types.

Due to upload size limitations, the images are stored in two zip files:

HAM10000_images_part1.zip (5000 JPEG files) Representing the Skin Cancer M.N.I.S.T.: HAM10000

HAM10000_images_part2.zip (5015 JPEG files) Representing the ISIC 2018 Challenge Training Set

The labels available in both datasets were as follows :  

**Skin Cancer Lesion Types**

| Lesion Type | Description                               |
|-------------|-------------------------------------------|
| AKIEC       | Actinic Keratosis / Intraepithelial Carcinoma |
| BCC         | Basal Cell Carcinoma                      |
| BKL         | Benign Keratosis-like Lesions             |
| DF          | Dermatofibroma                            |
| MEL         | Melanoma                                  |
| NV          | Melanocytic Nevi                          |
| VASC        | Vascular Lesions                          |

## CNN Architecture Design
**Pre-trained model selection :  DenseNet121**

Selecting the right pre-trained model or algorithm is crucial for achieving optimal performance in a specific problem domain. In the context of this project, DenseNet121 can be justified as a suitable choice for our algorithm for several reasons:

1. **Transfer Learning Capability:** DenseNet121 is a deep convolutional neural network (CNN) that has been pre-trained on a large-scale dataset (ImageNet). Transfer learning allows the model to leverage knowledge gained from this diverse dataset, capturing generic image features. This is particularly beneficial when working with medical images, as features learned from general images can be relevant to identifying patterns in skin lesions.

2. **Architectural Design:** DenseNet stands out for its unique architecture, where each layer receives feature maps from all preceding layers. This dense connectivity enhances feature reuse and information flow throughout the network, promoting better gradient flow during training. This architectural design often leads to improved convergence and better generalization, which is essential for accurate skin cancer classification.

3. **Performance in Medical Imaging:** DenseNet architectures have demonstrated success in various medical imaging tasks, showcasing their ability to extract relevant features from complex images. The model's ability to capture hierarchical and fine-grained patterns makes it a strong candidate for distinguishing between different types of skin lesions.

4. **Community Adoption and Benchmarking:** DenseNet121 is widely adopted and benchmarked in the machine learning community. Its performance has been well-documented in various image classification competitions and datasets. Leveraging a pre-trained model that has demonstrated success in similar tasks provides a strong foundation for building an effective skin cancer classification system.

DENSENET121 LAYER ARCHITECTURE
![Capture](https://github.com/Moyo-tech/DermDetect-AI/assets/80284832/47ff6717-4ab5-432a-bdf4-ac46782154ab)
