![Airbnb](images/airbnblogo.png)


# Airbnb-Superhost-Classifier
A classifier to determine if Superhost or Not SuperHost for Airbnb listings based on reviews and other features


### Objective

Reinvent the Superhost system by utilizing review data to identify Superhosts


### Key Business Insights
 
Identify what global travelers want  
Learn from Host / Guest interactions  
Monetize guest’s insights  
Identify Superhost traits  
Identify areas of improvement within airbnb’s product line  


### Modeling

Multinomial Naive Bayes  
Logistic Regression  
Random Forest  
Gradient Boosting  
LinearSVC  


### Results

Tf-idfVectorizer + Gradient Boosting Classifier + GridsearchCV 
86.7% Accuracy (9% + baseline model)



### Conclusion

Utilizing my best model (Tf-idfVectorizer + Gradient Boosting), I was able to make predictions on a hold out set. I validated the test results by checking them against ratings, reviews, response rate, activity, and superhost status. One key item that I took a deeper look into were the false positives. The false positives are hosts that are not labeled superhosts but the model identifies as superhost. These listing are important to take note of because they provide an opportunity for people looking for an awesome deal since they have great reviews, ratings, and typically are cheaper than their superhost counterparts. 

