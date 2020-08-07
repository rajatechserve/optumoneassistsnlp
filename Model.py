  import pickle
  import pandas as pd 
  
  df= pd.read_csv("data.csv", encoding="latin-1")
  # Features and Labels
 	df['label'] = df['Care'].map({'PCP': 0, 'Pharmacy': 1, 'Emergencey':2 })
    X = df['phrase']
 	y = df['label']
     
  # Extract Feature With CountVectorizer
  from sklearn.feature_extraction.text import CountVectorizer
 	cv = CountVectorizer()
 	X = cv.fit_transform(X) # Fit the Data
     
    pickle.dump(cv, open('fittranform.pkl', 'wb')) 
    
    from sklearn.model_selection import train_test_split
 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 	#Naive Bayes Classifier
 	from sklearn.naive_bayes import MultinomialNB

 	clf = MultinomialNB()
 	clf.fit(X_train,y_train)
 	clf.score(X_test,y_test)
    
    pickle.dump(clf, open('model.pkl', 'wb'))