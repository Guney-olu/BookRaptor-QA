#  RAG with RAPTOR indexing

[Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

## Installation

Before using RAPTOR, ensure Python 3.8+ is installed. Clone the BookRaptor-QA
repository and install necessary dependencies:

```bash
git clone https://github.com/Guney-olu/BookRaptor-QA.git
pip install -r requirements.txt
```

## Basic Usage (Build-in-rag-bot)

About it : Fusion of 3 Scifi Books 

```bash
cd chat
stremlit run app.py
```
## To test Retrieval
```bash
python chat/inference.py "Who lives on the mars" "demo_db/bookfusion.db"
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="demo.png">
  <img alt=" " src="demo.png" style="max-width: 60px; height: auto;">
</picture>

**Textbooks used for content extraction in folder books**

# Implemtation 
## Set up basic stuff 
Pdf Path?
Which embedding model to use?
Which model to use to summarize
Where to save the DB ?

Check the  Raptor/Implemetation.py to set up the stuff then run it 

```bash
python3 Raptor/Implemetation.py
```

## TODO

- [ ] **Create a Library** making Implementation.py more easy to use
- [ ] **Adding more models** trying and adding new models for summarizing
- [ ] **Improving Retrieval** Modifying the code for better Retrieval from the db
