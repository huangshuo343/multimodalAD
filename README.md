# multimodalAD
This is the code for the paper "Multi-stage Alignment and Fusion for Multi-modal Multi-class Alzheimer’s Disease Diagnosis" in conference the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2025).

To address the challenges in the alignment and fusion of data from heterogeneous and complex modalities, we propose here a novel framework for multimodal and multiclass AD diagnosis. Our framework integrates scalar images (T1-weighted MRI (T1w MRI), tau PET), high-dimensional fiber orientation distribution (FOD) from diffusion MRI, and tabular data (age, sex, Montreal Cognitive Assessment (MoCA) scores). First, we developed a SWIN-FOD model to process the complex 4D FODs efficiently. For fusing MRI and PET, we adapted the ALBEF model to handle 3D volumes. To capture relationships between features, we employed the pretrained priors in TabPFN, avoiding the need for additional feature alignment. Tested on the ADNI dataset (n = 1147), our model achieved 73.21\% accuracy, surpassing all comparison methods. Additionally, we analyzed the impact of each modality on the final diagnosis by Shapley analysis.

Figure 1 shows the workflow of mor method.


<img width="2895" height="1374" alt="Workflowgithub" src="https://github.com/user-attachments/assets/6df1909e-8952-466b-a1e1-5d5e38a5f594" />

We appreciate the code provided by 
[Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR), 
[ALBEF](https://github.com/salesforce/ALBEF), 
and [TabPFN](https://github.com/automl/TabPFN), as well as the Shapley analysis code provided by [SHAP](https://github.com/shap/shap). Part of the comparision on different image analysis methods uses the code in [Timm-3D](https://github.com/ZFTurbo/timm_3d).

If you find our work useful, please cite it as:

```bibtex
@InProceedings{HuangShuo_Multistage_MICCAI2025,
  author    = {Huang, Shuo and Zhong, Lujia and Shi, Yonggang},
  title     = {{Multistage Alignment and Fusion for Multimodal Multiclass Alzheimer’s Disease Diagnosis}},
  booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 15974},
  month     = {September},
  pages     = {375--385}
}
