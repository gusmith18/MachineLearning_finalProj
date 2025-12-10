# MachineLearning_finalProj

# Retriveing and cleaning data
https://www.kaggle.com/datasets/patrickgendotti/mtg-all-cards?resource=download

Download the data from this link and run data_clean.py to remove unnecessary items.

Then run image_clean_parallel.py to remove images of card backs from the data set, this might take a while.

Then run ability_preprocessing.py to add features.

Finally run add_color_data.py to add pixel percentages and normalized pixel percentages to the data. This might also take a while depending on internet connection and your computers specifications.

# Running the Maching Learning Models

After data cleaning is done, you can run either compare_NN_models.py to compare Neural Networks with different amounts of hidden layers or run predict_card_color_rf.py and visualize_card_color_rf.py to see the random forest model results.