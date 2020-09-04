This repository contains the code done for the Guided Research project.
The repository conatains two folders: Ingredient-Embedding and LSTM.
Ingredient Embedding contains code for generating the ingredient word embeddings using various NLP models such as Word2Vec and GloVe. The file GR_datapreprocessing contains the code for generating the results of the Word2Vec model considering the quantity of ingredients.
LSTM contains the code for training the LSTM: From LDA for generating recipe categories (LDA_RecipeTitles), to data-preprocessing (FoodCat2RecipeTitle_Mapping) to generating the pickle file needed for LSTM (LSTM_Trainer) and finally the LSTM model and it's results inclusive of model evaluation and t-SNE (Guided_LSTM).
In order to train the LSTM, please first create the pickle files from the LSTM_Trainer notebook and then use this in the Guided_LSTM notebook
All the files needed to execute the respective codes have been uploaded in the corresponding folders as well. Please upload these into the Google Colab Session before running the files. The code contains comments as to where which files will be needed.
Note: Layer1.json- Please download this from the Recipe1M+ dataset. Size: 1.8GB
In order to split the Recipe1M+ dataset into 500k dataset, command to be used in your remote terminal: split -l 500000 layer1.json
For further queries, please reach out to:

vindhya.singh@tum.de
chinmay.prabhakar@tum.de
