# Rice Image Classification Using Transfer Learning (GoogLeNet & AlexNet)

## Overview
This project demonstrates rice image classification using deep learning and transfer learning in MATLAB. It compares the performance of different neural network architectures (GoogLeNet, AlexNet, VGG16, VGG19) and training techniques (SGDM, ADAM, RMSprop) on a rice image dataset.

## Contents
- `googlenet3.m`: Transfer learning with GoogLeNet.
- `ricecnncw.m`: Transfer learning with AlexNet.
- `coursework.xlsx`: Contains detailed experiment results and comparisons.
- (Expected) `Rice_Image_Dataset/`: Folder containing subfolders for each rice class, with images.

## Dataset Structure
The dataset should be organized as follows:
```
Rice_Image_Dataset/
  Class1/
    img1.jpg
    img2.jpg
    ...
  Class2/
    img1.jpg
    ...
  ...
```

## How It Works
- **Data Preparation**: Images are loaded from `Rice_Image_Dataset`, split into training, validation, and test sets.
- **Transfer Learning**: The final layers of GoogLeNet or AlexNet are replaced to match the number of rice classes.
- **Data Augmentation**: Random reflections and translations are applied to improve model robustness.
- **Training**: The model is trained for 6 epochs with stochastic gradient descent (SGDM), ADAM, or RMSprop.
- **Evaluation**: Validation accuracy and elapsed time are reported for each run.

## Experiments & Results
A comprehensive set of experiments was conducted, varying the following:
- **Networks**: AlexNet, GoogLeNet, VGG16, VGG19
- **Training Techniques**: SGDM, ADAM, RMSprop
- **Data Augmentation & Hyperparameters**: Various settings for pixel range, learning rate, batch size, etc.

### Summary Table (from coursework.xlsx)
| Network   | Training | Avg. Validation Accuracy | Avg. Training Time | Max Validation Accuracy |
|-----------|----------|-------------------------|--------------------|------------------------|
| AlexNet   | SGDM     | 83%                     | 00:01:31           | 100%                   |
| AlexNet   | RMSprop  | 65%                     | 00:00:26           | 100%                   |
| AlexNet   | ADAM     | 89%                     | 00:01:02           | 100%                   |
| GoogLeNet | SGDM     | 85%                     | 00:00:16           | 100%                   |
| GoogLeNet | RMSprop  | 56%                     | 00:00:21           | 88%                    |
| GoogLeNet | ADAM     | 83%                     | 00:02:10           | 97%                    |
| VGG16     | SGDM     | 97%                     | 00:48:58           | 100%                   |
| VGG19     | SGDM     | 97%                     | 00:48:58           | 100%                   |

- **Best Results**: VGG16 and VGG19 with SGDM achieved the highest average validation accuracy (97%) but required the longest training time (~49 minutes).
- **Fastest Training**: GoogLeNet with SGDM trained the fastest (~16 seconds) with a strong average accuracy (85%).
- **Most Efficient**: AlexNet with ADAM provided a good balance of speed and accuracy (89% in ~1 minute).

## Usage
1. Place your rice image dataset in the `Rice_Image_Dataset` folder as described above.
2. Open either `googlenet3.m` or `ricecnncw.m` in MATLAB.
3. Run the script. The script will train and validate the model, printing accuracy and timing for each run.

## Requirements
- MATLAB with Deep Learning Toolbox
- Pre-trained networks: GoogLeNet, AlexNet, VGG16, VGG19 (downloadable via MATLAB Add-Ons)
- A dataset folder named `Rice_Image_Dataset` structured as above

## Notes
- The scripts use a very small percentage (0.1%) of images for validation and testing, which can be adjusted by changing the `percImgs` variable.
- Make sure the required pre-trained networks are installed in MATLAB.
- For detailed experiment settings and results, see `coursework.xlsx`.

## License
[Specify your license here] 