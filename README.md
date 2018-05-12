![Airbnb](images/airbnblogo.png)


# Airbnb Superhost Classifier
## Reinvent the Superhost system by utilizing review data to identify Superhosts

![title](images/title_slide.png)

### Objective

#### Create a classifier to determine if someone is a Superhost or not for Airbnb listings based on reviews

### Problem & Solution (Key Business Insights)
 
![problem_solution](images/problem_solution.png)

### Dataset

![dataset](images/dataset_info.png)

### Process & Setup

![process](images/setup_process.png)


### Modeling

![Modeling](images/models.png)


### Results

![final_model](images/final_model.png)

### Conclusion

Utilizing my best model (Tf-idfVectorizer + Gradient Boosting), I was able to make predictions on a hold out set. I validated the test results by checking them against ratings, reviews, response rate, activity, and superhost status. One key item that I took a deeper look into were the false positives. The false positives are hosts that are not labeled superhosts but the model identifies as superhost. These listing are important to take note of because they provide an opportunity for people looking for an awesome deal since they have great reviews, ratings, and typically are cheaper than their superhost counterparts. 


### Sources


* [insideAirbnb Dataset](http://insideairbnb.com/)
