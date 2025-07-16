# Progressive_Refinement_Attention_Feature_for_Joint_Source-Channel_Coding_in_Semantic_Image_Super-Resolution

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---
## 📖 Introduction  

**PRAF-JSCC** is a novel Joint_Source-Channel_Coding_framework designed for image super-resolution (SR) tasks in semantic communication.  

- **Why semantic SR?**  
  Traditional communication systems focus on bit-level accuracy, but semantic communication transmits *meaning* directly. In SR tasks, this allows the receiver to directly reconstruct high-resolution images without complex post-processing, which is crucial for applications like **telemedicine** and **satellite imaging**.  

- **Key challenges:**  
  1. Efficiently extracting deep semantic information from LR images for HR reconstruction.  
  2. Ensuring robust transmission under wireless channels.  

- **Our solution:**  
  ✅ **PRAF (Progressive Refinement Attention Feature)** module  
  - Progressive feature extraction for shallow & deep semantic features  
  - Dynamic SNR-aware channel attention to mitigate wireless channel noise  

  ✅ **End-to-end JSCC framework** for image SR  
  - Outperforms **Deep-JSCC** and traditional separated source-channel coding (SSCC)  

- **Performance:**  
  PRAF-JSCC achieves **better PSNR/SSIM** and more stable transmission across different SNRs, while maintaining **lower inference time** than existing methods.  


## 📰 Paper  

- **IEEE Wireless Communications Letters**  
- Volume **14**, Issue **7**, Pages **2099–2103**, July 2025  
- DOI: [10.1109/LWC.2025.3563231](https://doi.org/10.1109/LWC.2025.3563231)  

If you use this code, please cite:  

```bibtex
@ARTICLE{10973232,
  author={Huang, Zhen and Jia, Yunjian and Wen, Wanli and Liang, Liang and Yan, Jiping and Jiang, Nanlan},
  journal={IEEE Wireless Communications Letters}, 
  title={A Progressive Approach to Joint Source-Channel Coding for Image Super-Resolution Task in Semantic Communications}, 
  year={2025},
  volume={14},
  number={7},
  pages={2099-2103},
  doi={10.1109/LWC.2025.3563231}
}
