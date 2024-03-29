# MiniALBERT

Pre-trained Language Models (LMs) have become an integral part of Natural Language Processing (NLP) in recent years, due to their superior performance in downstream applications. In spite of this resounding success, the usability of LMs is constrained by computational and time complexity, along with their increasing size; an issue that has been referred to as `overparameterisation`. Different strategies have been proposed in the literature to alleviate these problems, with the aim to create effective compact models that nearly match the performance of their bloated counterparts with negligible performance losses. One of the most popular techniques in this area of research is model distillation.

Another potent but underutilised technique is cross-layer parameter sharing. In this work, we combine these two strategies and present **MiniALBERT**, a technique for converting the knowledge of fully parameterised LMs (such as BERT) into a compact recursive student. In addition, we investigate the application of bottleneck adapters for layer-wise adaptation of our recursive student, and also explore the efficacy of adapter tuning for fine-tuning of compact models. We test our proposed models on a number of general and biomedical NLP tasks to demonstrate their viability and compare them with the state-of-the-art and other existing compact models.

## Citation
```bibtex
@inproceedings{nouriborji2023minialbert,
  title={MiniALBERT: Model Distillation via Parameter-Efficient Recursive Transformers},
  author={Nouriborji, Mohammadmahdi and Rohanian, Omid and Kouchaki, Samaneh and Clifton, David A},
  booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  pages={1161--1173},
  year={2023}
}
```
