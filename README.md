# authentic-ai
Using machine learning to detect A.I generated essays.

## Problem
The rise of large language models (LLMs) has caused many folks to be concerned that LLMs will replace everyday human jobs. Specifically, educators are concerned that students may use LLMs to submit essays that are not their own. As a result, the students’ writing skills may deteriorate and their creative thinking ability may falter. In this project, I aim to tackle the following problem: how can we accurately assess whether a submitted essay was written by a large language model or written by a student?

## Problem Details
The problem is a classic binary classification problem (supervised learning) as the solution will simply classify an essay as A.I generated or authentic (student-written). To see how well my solution works, I will enter it into the [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) on Kaggle. This challenge evaluates solutions based on the Receiver Operator Curve (ROC) Area Under the Curve (AUC); hence, I will use the ROC AUC as my evaluation metric for my model. 

The Kaggle challenge already provides me with a dataset. However, the dataset is limited to 1378 essays of which 1375 are student written. Furthermore, the challenge only provides me essays from 2 prompts. The lack of data can be addressed by gathering more data from other data sources. Luckily, there are a variety of resources that can provide me with data that I can utilize in this project. Additionally, this problem is a problem that has garnered much attention from various researchers. Hence, I can utilize current research findings in building my solution. The papers I have utilized to guide my approach are cited in the References section. 

## References
This section outlines all the resources I used.

### Data
This section lists the data sources I used to build my dataset.

1. [daigt data - llama 70b and falcon 70b](https://www.kaggle.com/datasets/nbroad/daigt-data-llama-70b-and-falcon180b?select=falcon_180b_v1.csv)
2. [1000 Essays from Antrophic](https://www.kaggle.com/datasets/darraghdog/hello-claude-1000-essays-from-anthropic)
3. [LLM-generated essay using PaLM from Google Gen-AI](https://www.kaggle.com/datasets/kingki19/llm-generated-essay-using-palm-from-google-gen-ai)
4. [persuade corpus 2.0](https://www.kaggle.com/datasets/nbroad/persaude-corpus-2/?select=persuade_2.0_human_scores_demo_id_github.csv)
5. [DAIGT | External Dataset](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset)
6. [ArguGPT](https://www.kaggle.com/datasets/alejopaullier/argugpt?select=argugpt.csv)
7. [essays-with-instructions](https://huggingface.co/datasets/ChristophSchuhmann/essays-with-instructions)

### Papers
This section outlines all papers I used to guide my approach.

1. [ArguGPT: evaluating, understanding and identifying argumentative essays generated by GPT models](https://arxiv.org/abs/2304.07666)
2. [Generative AI Text Classification using Ensemble LLM Approaches](https://arxiv.org/pdf/2309.07755.pdf)
3. [Classification of Human-and AI-Generated Texts: Investigating Features for ChatGPT](https://arxiv.org/pdf/2308.05341.pdf)
4. [Will ChatGPT get you caught? Rethinking of Plagiarism Detection](https://arxiv.org/pdf/2302.04335.pdf)
5. [On the Possibilities of AI-Generated Text Detection](https://arxiv.org/abs/2304.04736)

## Author
If you have any questions about the project, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/jinalshah2002/)!
