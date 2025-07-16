# PRAF-JSCC
# PRAF-JSCC: Progressive Refinement Attention Feature for Joint Source-Channel Coding in Semantic Image Super-Resolution

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official implementation of the paper:  
> **A Progressive Approach to Joint Source-Channel Coding for Image Super-resolution Task in Semantic Communications**  
> *Zhen Huang, Yunjian Jia, Wanli Wen, Liang Liang, Jiping Yan, Nanlan Jiang*  
> IEEE Wireless Communications Letters, 2024.  

---

## 📖 Introduction  

This project provides the source code for **PRAF-JSCC**, a novel Joint Source-Channel Coding (JSCC) framework designed for **semantic image super-resolution (SR) tasks**.  

The image super-resolution (SR) task in semantic communication can directly apply the delivered information to the downstream SR task, eliminating complex processing at the receiver and significantly improving communication efficiency. This approach is vital for applications in areas such as telemedicine and satellite communications. Nevertheless, developing semantic communication systems for image SR tasks confronts challenges in creating high-performance joint source-channel coding (JSCC) schemes and mitigating wireless channel interference. In this paper, a progressive refinement attention feature (PRAF) module is proposed for the image SR task in semantic communication. This module effectively extracts deep semantic information from images via a progressive feature extraction strategy and adjusts the semantic information according to the SNRs using an improved channel attention mechanism. Building on PRAF, we custom-design the JSCC scheme for image SR tasks in semantic communications. Simulation results validate the effectiveness of the proposed PRAF module and confirm its superiority over existing deep neural networks (DNNs) based JSCC schemes and traditional separated source channel coding schemes.

---

## 📰 Paper  

> **A Progressive Approach to Joint Source-Channel Coding for Image Super-resolution Task in Semantic Communications**  
> IEEE Wireless Communications Letters, 2024  
> [PDF Download](./WCL2024-2585_Proof_hi.pdf)  

If you use this code, please cite:  

```bibtex
@article{huang2024prafjscc,
  title={A Progressive Approach to Joint Source-Channel Coding for Image Super-resolution Task in Semantic Communications},
  author={Huang, Zhen and Jia, Yunjian and Wen, Wanli and Liang, Liang and Yan, Jiping and Jiang, Nanlan},
  journal={IEEE Wireless Communications Letters},
  year={2024}
}
