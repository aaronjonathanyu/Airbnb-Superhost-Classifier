![title](images/title.png)

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

### Feature Importance

![feature_importance](images/feature_importance.png)

### False Positive Example

This false positive is an example of others who are missed by airbnb's current superhost system. This person below has 5+ star rating, 63+ great reviews, and has priced her listing considerably lower than her superhost counterparts within the same neighborhood. She would be counted as a great deal for bargain hunters looking for an airbnb to stay at in New York. The other business insight from this specific listing is that we can reach out to her, offering pointers on how to get her to superhost status. Superhosts typically get featured ads on airbnb's website, generate more revenue, and get special perks such as coupon certificates to use at other airbnb listings.

![false_positive_example](images/false_positive.png)

### Conclusion

Utilizing my best model (Tf-idfVectorizer + Gradient Boosting), I was able to make predictions on a hold out set. I validated the test results by checking them against ratings, reviews, response rate, activity, and superhost status. One key item that I took a deeper look into were the false positives. The false positives are hosts that are not labeled superhosts but the model identifies as superhost. These listing are important to take note of because they offer an opportunity for people looking for an awesome deal since they have great reviews, ratings, and typically are cheaper than their superhost counterparts. 


### Sources


* [insideAirbnb Dataset](http://insideairbnb.com/)
