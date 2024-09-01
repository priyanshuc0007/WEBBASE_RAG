from setuptools import setup, find_packages

setup(
    name='chatbot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'openai',
        'faiss-cpu',
        'nltk'
    ],
    description='A package to load, split, embed, and store text data using RAG',
    author='priyanshu chauhan',
    author_email='priyanshuc111@gmail.com',
)
