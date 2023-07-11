[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Parrot-GPT

This repository contains a Jupyter Notebook that allows users to locally run a GPT model on a Mac M1. The notebook utilizes the GPT-Neo model from Hugging Face, specifically the 2.7 billion parameter version. The model can be run on a GPU if available, but it is not mandatory.

## Installation

To run the notebook, please follow these steps:

1. Clone the repository to your local machine.

2. Create a Conda environment using the provided `environment.yml` file: 
    ```
    conda env create -f environment.yml
    ```

3. Activate the Conda environment: conda activate parrot-gpt

4. Install the Jupyter kernel for the notebook:
    ```
    python -m ipykernel install --user --name gpt_vid
    ```

5. Start Jupyter Notebook in the repository directory: 
    ```
    jupyter notebook
    ```

6. In the Jupyter Notebook interface, open the notebook file (parrot-gpt.ipynb) and select the gpt_vid kernel.

7. Follow the instructions in the notebook to enter a prompt and specify the desired length of the generated output.

8. Enjoy interacting with the Parrot-GPT model!

## Model Details

The GPT-Neo model used in this repository is the EleutherAI/gpt-neo-2.7B version, which consists of 2.7 billion parameters. If you prefer a lighter model, you can also use the EleutherAI/gpt-neo-1.3B version, which has 1.3 billion parameters. The choice of model depends on the available GPU VRAM, as the 2.7B model requires approximately 13 GB, while the 1.3B model requires approximately 7.5 GB.

## Acknowledgments
I would like to extend my special thanks to the following resources:

- The YouTube video [Deep Dive into GPT-3 and GPT-Neo](https://www.youtube.com/watch?v=d_ypajqmwcU&t=7s) provided a comprehensive overview and detailed explanations of GPT-3 and GPT-Neo, which greatly enhanced our understanding of these models.

- [GPTNeo_notebook](https://github.com/mallorbc/GPTNeo_notebook) GitHub repository, created by [mallorbc](https://github.com/mallorbc). This repository provided valuable insights and examples that greatly aided in the development of our project.

- [gpt-neo](https://github.com/EleutherAI/gpt-neo) GitHub repository, developed by the team at [EleutherAI](https://github.com/EleutherAI). The groundbreaking work done by this project was instrumental in advancing the field of language models and served as a significant inspiration for our own endeavors.

The GPT-Neo model is developed by EleutherAI.
This project utilizes the Hugging Face library for easy access to Transformer models.
Special thanks to the developers and contributors of PyTorch and Jupyter Notebook for their invaluable tools.
Disclaimer
Please note that the generated text may not always provide accurate or reliable information. The model's responses are based on patterns learned from large amounts of text data and should be used with caution. Use the generated output for informational purposes only and exercise critical thinking when interpreting the results.

## License

This project is licensed under the MIT License.

Feel free to explore, modify, and use the code as per the terms of the license.
