# FewshotQA: A simple framework for few-shot learning of question answering tasks using pre-trained text-to-text models

Making the repository of [Amazon Science ](https://github.com/amazon-science/fewshotqa) more accessible to the user.


### Install prerequisits:
Get the experiment files.

```
$ bash setup.sh
```

Please install PyTorch seperatly based on the GPU or CPU you want to use.
We optimize this repository based on the M1 Arm.

### Cite authors

```
@inproceedings{chada-natarajan-2021-fewshotqa,
    title = "{F}ewshot{QA}: A simple framework for few-shot learning of question answering tasks using pre-trained text-to-text models",
    author = "Chada, Rakesh  and
      Natarajan, Pradeep",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.491",
    doi = "10.18653/v1/2021.emnlp-main.491",
    pages = "6081--6090",
    abstract = "The task of learning from only a few examples (called a few-shot setting) is of key importance and relevance to a real-world setting. For question answering (QA), the current state-of-the-art pre-trained models typically need fine-tuning on tens of thousands of examples to obtain good results. Their performance degrades significantly in a few-shot setting ({\textless} 100 examples). To address this, we propose a simple fine-tuning framework that leverages pre-trained text-to-text models and is directly aligned with their pre-training framework. Specifically, we construct the input as a concatenation of the question, a mask token representing the answer span and a context. Given this input, the model is fine-tuned using the same objective as that of its pre-training objective. Through experimental studies on various few-shot configurations, we show that this formulation leads to significant gains on multiple QA benchmarks (an absolute gain of 34.2 F1 points on average when there are only 16 training examples). The gains extend further when used with larger models (Eg:- 72.3 F1 on SQuAD using BART-large with only 32 examples) and translate well to a multilingual setting . On the multilingual TydiQA benchmark, our model outperforms the XLM-Roberta-large by an absolute margin of upto 40 F1 points and an average of 33 F1 points in a few-shot setting ({\textless}= 64 training examples). We conduct detailed ablation studies to analyze factors contributing to these gains.",
}
```
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
