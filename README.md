# gan-ultrasound-augmentation

Related Paper: Zaman, A., Park, S.H., Bang, H. et al. Generative approach for data augmentation for deep learning-based bone surface segmentation from ultrasound images. Int J CARS 15, 931–941 (2020). https://doi.org/10.1007/s11548-020-02192-1

### Paper Abstract
#### Purpose
Precise localization of cystic bone lesions is crucial for osteolytic bone tumor surgery. Recently, there is a move toward ultrasound imaging over plain radiographs (X-rays) for intra-operative navigation due to the radiation-free and cost-effectiveness of the modality. In this process, the intra-operative bone model reconstructed from the segmented ultrasound image is registered with the pre-operative bone model. Deep learning approaches have recently shown remarkable success in bone surface segmentation from ultrasound images. However, to train deep learning models effectively with limited dataset size, data augmentation is essential. This study investigates the applicability of the generative approach for data augmentation as well as identifies standard data augmentation approaches for bone surface segmentation from ultrasound images.

#### Methods
The generative approach we used in our work is based on Pix2Pix image-to-image translation network. We have proposed a multiple-snapshot approach, which mitigates the uni-modal deterministic output issue in the Pix2Pix network without using any complex architecture and training process. We also identified standard data augmentation approaches necessary for ultrasound bone surface segmentation through experiments.

#### Results
We have evaluated our networks using 800 ultrasound images from trained regions (humerus bone) and 1200 ultrasound images from untrained regions (tibia and femur bones) using four different augmentation approaches. The results show that the generative augmentation approach has a positive impact on accuracy in both trained (+ 4.88%) and untrained regions (+ 25.84%) compared to using only standard augmentations. Moreover, compared to standard augmentation approaches, the addition of the generative augmentation approach also showed a similar trend in both trained (+ 8.74%) and untrained (+ 11.55%) regions.

#### Conclusion
Generative approaches are very beneficial for data augmentation, where limited dataset size is prevalent, such as ultrasound bone segmentation. The proposed multiple-snapshot Pix2Pix approach has the potential to generate multimodal images, which enlarges the dataset considerably.

## High Level Overview
This code was used to train and test the model used in the paper. The image files have been annonimized to protect the volunteer information. The ground truths are created by in-house domain experts. 
