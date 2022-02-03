# SkimLit
An NLP model to classify abstract sentences into the role they play (e.g. objective, methods, results, etc..) to enable researchers to skim through the literature and dive deeper when necessary.

Try Demo; **[WEB APP](https://huggingface.co/spaces/Vrk/SkimLit)**

<img src="images/app.png" width=80% height=80%>

* **More specificially, I'am going to replicate the deep learning model behind the 2017 paper [*PubMed 200k RCT: a Dataset for Sequenctial Sentence Classification in Medical Abstracts*](https://arxiv.org/abs/1710.06071).**

## Dataset Used
[PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)

* The PubMed 200k RCT dataset is described in *Franck Dernoncourt, Ji Young Lee. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071). International Joint Conference on Natural Language Processing (IJCNLP). 2017.*

Some miscellaneous information:
- PubMed 20k is a subset of PubMed 200k. I.e., any abstract present in PubMed 20k is also present in PubMed 200k. 
- `PubMed_200k_RCT` is the same as `PubMed_200k_RCT_numbers_replaced_with_at_sign`, except that in the latter all numbers had been replaced by `@`. (same for `PubMed_20k_RCT` vs. `PubMed_20k_RCT_numbers_replaced_with_at_sign`).

- **Count Plot**
<img src="https://user-images.githubusercontent.com/59719046/138639626-48336732-ca8f-4bfe-8063-0e1f7a7c6ae6.png" width=50% height=50%>

## Models Tried
All the note books are availabel [here]("https://github.com/vishalrk1/SkimLit/tree/main/Notebooks")

- NaiveBiase Model -> 72% Accuracy
- Conv1D Model -> 78% Accuracy
- Model using pretrained token embedding ( Universal sentence embedding ) -> 75% Accuracy
- Conv1D Model using character level embedding -> 73% Accuracy 
- Model with both token and charcter level embedding -> 76% Accuracy
- Model with token, character and position level embedding ( https://arxiv.org/pdf/1612.05251.pdf ) -> 81% Accuracy
<!--         <img src="https://user-images.githubusercontent.com/59719046/138639849-8bb0dcb4-f307-45cf-82ec-671530680863.png" width=40% height=40%> -->
- Model described in [this](https://arxiv.org/pdf/1612.05251.pdf) paper with bert embedding -> 88% Accuracy
## Final Results

### **Results of all Models**
<img src="images/modeling results.png" width=80% height=80%>

### **Best Performong Model**

<img src="images/bert model.png" width=100% height=100%>

### **Final Outputs** 

<img src="images/Streamlit.png" width=100% height=100%>


## Packages Used
- Tensorflow
- tensorflow_text
- tensorflow_hub
- sklearn
- Matplotlib
- numpy
- pandas
- spaCy


## Contact Me


<p align="start">
    <a href="https://github.com/vishalrk1" target="_blank">
        <img alt="Github" src="https://img.shields.io/badge/Github-%23F37626.svg?style=for-the-badge&logo=github&logoColor=white" />&nbsp;
    </a>
<!--     <a href="https://twitter.com/ArizArmeidi" target="_blank">
        <img src="https://img.shields.io/badge/-Twitter-2CA5E0?logo=twitter&style=for-the-badge&logoColor=white&color=black" alt="Twitter" />
    </a> -->
    <a href="https://www.linkedin.com/in/vishal-karangale-126492216/" target="_blank">
        <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-%23F37626.svg?style=for-the-badge&logo=linkedin&logoColor=white" />&nbsp;
    </a>
     <a href="https://www.instagram.com/vishal_rk1/" target="_blank">
       <img alt="Instagram" src="https://img.shields.io/badge/Instagram-%23F37626.svg?style=for-the-badge&logo=instagram&logoColor=white" />&nbsp;
    </a>
</p>
