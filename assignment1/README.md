# Assignment 1

## Overview

Assignment 1 includes three parts:
1. KNN
2. Linear Regression
3. Logistic Regression

File structure:
```
assignment1
├── README.md  (this file)
├── algorithms  (source code folder)
├── *.ipynb  (notebooks)
├── get_datasets.py  (download script)
└── dataset (datasets folder)
```

## Prepare Datasets

Before you start, you need to run the following command (in terminal or in notebook beginning with `!` ) to download the dataset:

```
# This command will download the dataset and put it in "./dataset".
python get_dataset.py
```

## Implementation

You should run all code blocks in the following jupyter notebooks and write your answers to all inline questions included in the notebooks:

1. `assignment_knn.ipynb`
2. `assignmnet_linear.ipynb`
3. `assignment_logistic.ipynb`


Go through the notebooks to understand the structure. These notebooks will require you to complete the following implementations:

1. `algorithms/knn.py`: Implement KNN algorithm
2. `algorithms/linear_regression.py`: Implement linear regression algorithm
3. `algorithms/logistic_regression.py`: Implement logistic regression algorithm

## Submission

After complete all the functions and questions, you should upload the following files to Gradescope:

1. All source files in "algorithms" directory.
2. All 3 notebooks (`.ipynb` files)
3. Exported PDF of notebooks(You could export Jupyter notebook as PDF in web portal, do not need to install LaTeX packages) and merge them

You should organize files like this:
1. Put all source files(three python files and three notebooks) into a single `.zip` file then upload it on Gradescope
2. Merge exported PDF of notebooks into a single PDF file and upload it on Gradescope

# Short Instruction About How to Use Jupyter Notebook

Part of our skeleton code is provided using Jupyter notebook("assignment_XX.ipynb"). So there we provide a short instruction about how to use Jupyter Notebook.

Jupyter notebook is a powerfull interactive developement tool. Here we could Jupyter notebook file as notebook for short.

Jupyter notebook is pre-installed on Datahub, you can just use it via web-portal. You could also install it on your local machine. Here is the official install instruction([https://jupyter.org/install](https://jupyter.org/install)), which also included how to run it via terminal.

You may find on Jupyter website, they provide Jupyter Lab. Jupyter Lab is basically a new version of Jupyter notebook with more features and different interface. The basic usage is almost the same.

All notebooks are made up with multiple blocks. There are different kinds of blocks, the most common blocks are:
* Code block
* Markdown block

(You can change the category of blocks in web portal anytime you want)

## Code

For code block, you can write python code in code block, after finishing your code you could press run button on the jupyter note interface(normally on the top of the web interface). You can also use `Ctrl + Enter` or `Shift + Enter` to execute the block.

After you execute a block, Jupyter Notebook will execute your code with python and store all the function and variable you defined in memory. So you could still use those variables and function is other blocks.

For code blocks, you can think of jupyter notebook as a python console with an interface.

## Markdown

Markdown block is where you can write some text.

When you execute Markdown block(the same method as Code block), your text will be compiled using Markdown grammar, instead of executing it with python.

In our assignment, we put some inline questions in notebooks, you are supposed to answer them with text.

## Conclusion

In all, notebooks are basially some combination of code and text.

## Reference

We provide some other usefull links here for your reference.

* Official tutorials
    * https://mybinder.org/v2/gh/ipython/ipython-in-depth/master?filepath=binder/Index.ipynb
    * https://mybinder.org/v2/gh/jupyterlab/jupyterlab-demo/master?urlpath=lab/tree/demo
* Some online tutorials
    * https://www.dataquest.io/blog/jupyter-notebook-tutorial/

## How to Set Up Python Environment (Optional)

If you run the assignment on Datahub, you do **not** need to set up python environment. Our code is tested on Datahub.

If you run the assignment on your local machine or other environment and meet some problems, you **may** need to set up python environment.

We prepare a `requirements.txt` file (same versions as the datahub packages) for you to install python packages. You can install the packages by running the following command in terminal:

```sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instruction on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

```sh
conda create -n ece176 python=3.9.5  # same python version as the datahub
conda activate ece176
pip install -r requirements.txt
```

If you have any questions, feel free to contact TAs.

### Other Tips

How to unzip a zip file:

```sh
unzip XX.zip  # in terminal or in notebook beginning with `!`
```
