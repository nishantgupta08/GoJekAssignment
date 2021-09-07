
import sys
sys.path.append('../../')
import os
import pandas as pd
import numpy as np
import pickle
import hdbscan
from haversine import haversine
# from src.utils.time import robust_hour_of_iso_date,day_of_iso_date,month_of_iso_date
from IPython.terminal.debugger import set_trace as keyboard
from src.utils.store import AssignmentStore

# PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    ''' Used Haverisne distance        
    '''
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creating temporal features like day of week and is_busy_hour
    '''
    store = AssignmentStore()
    # df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    # df["event_day"] = df["event_timestamp"].apply(day_of_iso_date)
    # df["event_month"] = df["event_timestamp"].apply(month_of_iso_date)

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], infer_datetime_format=True).dt.tz_convert('Asia/Kolkata')
    
    df["date"] = df["event_timestamp"].dt.date
    df["hourofday"] = df["event_timestamp"].dt.hour
    df["dayofweek"] = df["event_timestamp"].dt.dayofweek
    
    df["is_busy_hour"] = df["hourofday"].apply(lambda x:  1 if x>=5 and x<=15 else 0)
    
    # Entropy based binning for hourofday keeping in mind the target variable acceptance rate

    # filepath = os.path.join(PROJECT_DIR, "models//discretiser.pkl")
    # discretiser = pickle.load(open(filepath, "rb" ))
    
    discretiser = store.get_model("discretiser.pkl")
    df["hourly_bins"] = discretiser.transform(np.array(df['hourofday']).reshape(-1, 1))
    
    return df   

def get_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return [x,y,z]

def geographical_features(df:pd.DataFrame) -> pd.DataFrame:
    '''Used HDBSCAN clustering algorithm as KMeans doesn't work well for geosptial 
       data reason being non linear separability.
       I have kept both the driver and customer cluster information so that it learns the pattern 
       if for some particular pickup clusters there could be more acceptance rate and 
       similarly driver being at some particular clusters could have more  acceptance rate. 
    '''
    store = AssignmentStore()
    
    
    df_cord_customer = df[["pickup_latitude", "pickup_longitude"]]
    df_cord_driver = df[["driver_latitude", "driver_longitude"]]
    

    # Computed on another machine 
    # model_customer = pickle.load(open("models//clusterer_customer.sav", "rb" ))
    # cluster_customer_labels,probabilities = hdbscan.approximate_predict(model_customer,np.radians(df_cord_customer))
    # cluster_labels_train_server = store.get_processed("cluster_labels_train_server.csv")
    # df["cluster_customer_labels"] = cluster_labels_train_server["customer_labels"]
    
    # Computed on another machine 
    # cluster_driver_labels,probabilities = hdbscan.approximate_predict(model_driver,df_cord_driver)
    # cluster_labels_train_server = store.get_processed("cluster_labels_train_server.csv")
    # df["cluster_driver_labels"] = cluster_labels_train_server["driver_labels"]

    model = store.get_model("kmeanModel.pkl")
    df_cord_driver_cartesian = pd.DataFrame(df_cord_driver.apply(lambda x: get_cartesian(x["driver_latitude"],x["driver_longitude"]),axis=1).tolist())
    df_cord_customer_cartesian = pd.DataFrame(df_cord_customer.apply(lambda x: get_cartesian(x["pickup_latitude"],x["pickup_longitude"]),axis=1).tolist())    
    df["driver_cluster_label"] = model.predict(df_cord_driver_cartesian)
    df["customer_cluster_label"] = model.predict(df_cord_customer_cartesian)
    
    return df

def participant_general_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Created features that can capture variance at both driver and timestamp level
    '''
    store = AssignmentStore()
    # Taking in to factor both  mean and variance of a driver 
    dayofweek_features = store.get_processed("dayofweek_features_train.csv")
    hourofday_features = store.get_processed("hourofday_features_train.csv")
    last5day_features_train = store.get_processed("last5day_features_train.csv")

    last5day_features_train_group = last5day_features_train.groupby("driver_id")[["driver_id","is_accepted_last5day_rolling_mean"]].tail(1).reset_index() 
    
    df = df.merge(last5day_features_train_group[["driver_id","is_accepted_last5day_rolling_mean"]],on=["driver_id"],how="left")
    # To make merge faster 
    # dayofweek_features.set_index(['driver_id','dayofweek'].stack().unstack([1,2], inplace=True)
    # x = df.join(dayofweek_features['dayofweek_average'], how='left')
    
    df = df.merge(dayofweek_features[["driver_id","dayofweek","driver_dayofweek_average"]],on=["driver_id","dayofweek"],how="left")\
        .merge(dayofweek_features[["dayofweek","dayofweek_average"]].drop_duplicates(),on=["dayofweek"],how="left")
    
    df = df.merge(hourofday_features[["driver_id","hourofday","driver_hourofday_average"]],on=["driver_id","hourofday"],how="left")\
           .merge(hourofday_features[["hourofday","hourofday_average"]].drop_duplicates(),on=["hourofday"],how="left")

    df["driver_dayofweek_average"].fillna(df["dayofweek_average"],inplace = True)
    df["driver_hourofday_average"].fillna(df["hourofday_average"],inplace = True)
    df["is_accepted_last5day_rolling_mean"].fillna(df["is_accepted_last5day_rolling_mean"].mean(),inplace = True)
    return df

def participant_mean_acceptance_time(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creating Features to understand how quick or slow does driver accept it 
    '''
    return df