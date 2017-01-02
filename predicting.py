import pickle
import pandas as pd

new_data_csv='new_data.csv'
selected_features = ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge',
                     'surge_pct', 'trips_in_first_30_days', 'luxury_car_user',
                     'weekday_pct', 'city_Astapor', 'city_King\'s Landing', 'city_Winterfell',
                     'phone_Android', 'phone_iPhone', 'phone_no_phone', 'signup_dow_0',
                     'signup_dow_1', 'signup_dow_2', 'signup_dow_3', 'signup_dow_4',
                     'signup_dow_5', 'signup_dow_6']
target = 'churn'


def load_cleaned_data():
    df = pd.read_csv(new_data_csv)
    X = df[selected_features].values
    y = df[target].values

    return df,X, y

if __name__=='__main__':

    #load cleaned data
    df,X,y=load_cleaned_data()

    #load pre-trained model
    filename='LogisticRegression.pkl'
    with open(filename,'r') as handle:
        input_model=handle.read()
    model=pickle.loads(input_model)

    #make prediction for each data point
    for i, data_point in enumerate(X):
        print "For the {} th new rider,".format(i),\
        "Predicted value:",model.predict(data_point.reshape(1,X.shape[1]))[0],",",\
        "Actual value",y[i]