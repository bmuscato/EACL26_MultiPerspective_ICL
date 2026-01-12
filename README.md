# This repository is the official implementation of the EACL 2026 Paper entitled "Seeing All Sides: Multi-Perspective In-Context Learning for Subjective NLP" 

## Abstract

Modern language models excel at factual reasoning but struggle with value diversity: the multiplicity of plausible human perspectives. 
Tasks such as hate speech or sexism detection expose this limitation, where human disagreement captures the diversity of perspectives that models need to account for, rather than dataset noise. In this paper, we explore whether multi-perspective in-context learning (ICL) can align large language models (LLMs) with this diversity without parameter updates. 
We evaluate four LLMs on five datasets across three languages (English, Arabic, Italian), considering three label-space representations (aggregated hard, disaggregated hard, and disaggregated soft) and five demonstration selection and ordering strategies. Our multi-perspective approach outperforms standard prompting on aggregated English labels, while disaggregated soft predictions better align with human judgments in Arabic and Italian datasets.These findings highlight the importance of perspective-aware LLMs for reducing bias and polarization, while also revealing the challenges of applying ICL to socially sensitive tasks.We further probe the model faithfulness using XAI, offering insights into how LLMs handle human disagreement.


If you find this code useful in your work, please cite our paper: 
