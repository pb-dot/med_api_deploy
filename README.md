First Go to the helpers folder to run the gen1.py and gen2.py to generate the model.pkl files<br>
ie SVM model and KNN model with tfidf text representation in gen1.py<br>
and LR model which uses transformer as text representation in gen2.py<br>

The util.py has hepler function needed by app.py for prediction and text preprocessing<br>

While deploying remove the helpers folder as we already have .pkl files<br>

### Run the whole application using<br>
pip install -r requirements.txt --user <br>
streamlit run app.py
