> Em desenvolvimento

Este repositório contém todo o código-fonte do curso __Manual Prático do Deep Learning__ no Udemy (link em breve). Siga as instruções abaixo para instalação e configuração do repositório.

# Instalação

1. Baixe ou clone o repositório.
2. Baixe e instale o [Miniconda](https://conda.io/miniconda.html).
3. Abra o terminal e vá para a pasta do repositório.
4. Siga as instruções abaixo de acordo com o seu sistema operacional:
    - No __Windows__
        - Para instalar o ambiente:
            ```sh
            $ conda env create -f dl_win.yml
            ```
        - Para ativar o ambiente: 
            ```sh
            $ activate ml
            ```
    - No __Linux/Mac__:
        - Para instalar o ambiente:
            ```sh
            $ conda env create -f dl_unix.yml
            ```
        - Para ativar o ambiente: 
            ```sh
            $ source activate ml
            ```
5. Execute o Jupyter Notebook:
    ```sh
    $ jupyter notebook
    ```
> A instalação do ambiente deve ser executada somente uma vez, enquanto o ambiente deve ser ativado sempre que você quiser executar os códigos do repositório.

Se você preferir instalar todos os pacotes individualmente, digite:
```sh
$ conda create -n dl python=3.6 numpy pandas matplotlib jupyter scikit-learn widgetsnbextension
$ conda install -c conda-forge jupyter_contrib_nbextensions
```