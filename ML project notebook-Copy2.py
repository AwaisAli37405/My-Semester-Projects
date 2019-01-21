
# coding: utf-8

# # SVR Version (Normalized)

# # Importing Libraries

# In[1]:

import pandas as pd
import numpy as np
import itertools 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
np.set_printoptions(threshold=np.nan)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# # Reading the Dataset

# In[2]:

airlines_data=pd.read_csv("Dataset/airlines.csv")
airlines_data.columns=['IATA_CODE','AIRLINE']

print(airlines_data)


# In[3]:

flights_data = pd.read_csv("Dataset/flights.csv")

#flights_data.columns =  ['year','month','day','day_of_week','IATA_CODE','flight_num','tail_num','origin_airport','dest_airport','sched_dep','dep_time','dep_delay','taxi_out','wheels_off','sched_time','elapsed_time','air_time','distance','wheels_on','taxi_in','sched_arrival','arrival_time','arrival_delay','diverted','cancelled','cancellation_reason','air_sys_delay','security_delay','airline_delay','late_aircraft_delay','weather_delay']

#     pd_data[columns].replace('',np.nan,inplace=True)

#     pd_data.dropna()

# data = np.array(pd_data[['year','month','day','day_of_week','IATA_CODE','flight_num','tail_num','origin_airport','dest_airport','sched_dep','dep_time','dep_delay','taxi_out','wheels_off','sched_time','elapsed_time','air_time','distance','wheels_on','taxi_in','sched_arrival','arrival_time','arrival_delay','diverted','cancelled']].dropna(how='all'))
# Nan_data = np.array(pd_data[['air_sys_delay','security_delay','airline_delay','late_aircraft_delay','weather_delay','cancellation_reason']].dropna(how='all'))

flights_data = pd.merge(flights_data, airlines_data, left_on='AIRLINE', right_on='IATA_CODE', how='left')
flights_data.drop('IATA_CODE', axis=1,inplace=True)
flights_data.drop('AIRLINE_x', axis=1,inplace=True)
flights_data.rename(columns={'AIRLINE_y': 'AIRLINE'}, inplace=True)
    


# In[4]:


variables_to_remove = [ 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY',
       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
       'WEATHER_DELAY']
flights_data.drop(variables_to_remove, axis = 1, inplace = True)


# In[5]:

print(flights_data.columns)
missing_df = flights_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(flights_data.shape[0]-missing_df['missing values'])/flights_data.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)


# In[7]:

flights_data.dropna(inplace=True)#removing nan's from the data frame

for col in flights_data:
    print ("%d NULL values are found in column %s" % (flights_data[col].isnull().sum().sum(), col))
    


# In[8]:

#missing_df = flights_data.isnull().sum(axis=0).reset_index()
# missing_df.columns = ['variable', 'missing values']
# missing_df['filling factor (%)']=(flights_data.shape[0]-missing_df['missing values'])/flights_data.shape[0]*100
# missing_df.sort_values('filling factor (%)').reset_index(drop = True)

flights_data.loc[:20, ['WHEELS_OFF','TAXI_OUT']]


# # Now converting the initial data frame dates to actual datetime format.

# In[9]:

#_________________________________________________________
# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)


# In[10]:

import datetime
flights_data['DATE'] = pd.to_datetime(flights_data[['YEAR','MONTH', 'DAY']])

flights_data['SCHEDULED_DEPARTURE'] = create_flight_time(flights_data, 'SCHEDULED_DEPARTURE')
flights_data['DEPARTURE_TIME'] = flights_data['DEPARTURE_TIME'].apply(format_heure)
flights_data['SCHEDULED_ARRIVAL'] = flights_data['SCHEDULED_ARRIVAL'].apply(format_heure)
flights_data['ARRIVAL_TIME'] = flights_data['ARRIVAL_TIME'].apply(format_heure)
# flights_data['WHEELS_OFF'] = flights_data['WHEELS_OFF'].apply(format_heure)
# flights_data['TAXI_OUT'] = flights_data['TAXI_OUT'].apply(format_heure)
#__________________________________________________________________________


# In[11]:

flights_data.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']].loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]


# In[14]:

flights_data=flights_data[flights_data.SCHEDULED_DEPARTURE.notnull()]


# In[15]:





fct = lambda x:x.hour*3600+x.minute*60+x.second
flights_data['SCHEDULED_DEPARTURE']=flights_data['SCHEDULED_DEPARTURE'].apply(lambda x:x.time())
flights_data['SCHEDULED_DEPARTURE']=flights_data['SCHEDULED_DEPARTURE'].apply(fct)

#flights_data['SCHEDULED_ARRIVAL']=flights_data['SCHEDULED_ARRIVAL'].apply(lambda x:x.time())
flights_data['SCHEDULED_ARRIVAL']=flights_data['SCHEDULED_ARRIVAL'].apply(fct)

#flights_data['DEPARTURE_TIME']=flights_data['DEPARTURE_TIME'].apply(lambda x:x.time())
flights_data['DEPARTURE_TIME']=flights_data['DEPARTURE_TIME'].apply(fct)

#flights_data['ARRIVAL_TIME']=flights_data['ARRIVAL_TIME'].apply(lambda x:x.time())
flights_data['ARRIVAL_TIME']=flights_data['ARRIVAL_TIME'].apply(fct)

# flights_data['WHEELS_OFF']=flights_data['WHEELS_OFF'].apply(fct)

# flights_data['TAXI_OUT']=flights_data['TAXI_OUT'].apply(fct)

flights_data.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',
             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]


# In[16]:

#for the label departure delay

# departure_features = np.array(flights_data[['AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT']].dropna(axis=1,how='all'))
# label_departure_delay = np.array(flights_data[['DEPARTURE_DELAY']].dropna(axis=1,how='all'))
# print(departure_features.shape,label_departure_delay.shape)




departure_features = np.array(flights_data[['FLIGHT_NUMBER', 'TAIL_NUMBER',
       'ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE',
       'DEPARTURE_TIME', 'WHEELS_OFF',
       'AIRLINE']].dropna(axis=1,how='all'))
label_departure_delay = np.array(flights_data[['DEPARTURE_DELAY']].dropna(axis=1,how='all'))
print(departure_features.shape,label_departure_delay.shape)


# In[17]:


# # print(departure_features[0:10,:])
# # departure_features[:5714008,3]=departure_features[:5714008,3].astype(str)  

unique,count=np.unique(departure_features[:,1],return_counts=True)
# print(unique,count)
# # from scipy.stats import itemfreq
# # itemfreq(departure_features[:,3])



# In[18]:

#removing rows from origin airport having integer values


print(len(departure_features))


#for the airline
count=0
masked_index=[]
for i in range(len(departure_features)):
    #or k in range(departure_features.shape[1]):
    if(isinstance(departure_features[i,6],(str))==False):
        
#         print("df",i," =",departure_features[i,3])
        count+=1
        masked_index.append(i)
        #print ("row",i)
            #print(X[i],Y[i])
print ("length of masked=",len(masked_index))
print("number of examples having noise:",count)

departure_features=np.delete(departure_features,masked_index,axis=0)
label_departure_delay=np.delete(label_departure_delay,masked_index,axis=0)
print ("for airline",departure_features.shape,label_departure_delay.shape)

#for tail number
count=0
masked_index=[]
for i in range(len(departure_features)):
    #or k in range(departure_features.shape[1]):
    if(isinstance(departure_features[i,2],(str))==False):
        
#         print("df",i," =",departure_features[i,3])
        count+=1
        masked_index.append(i)
        #print ("row",i)
            #print(X[i],Y[i])
print ("length of masked=",len(masked_index))
print("number of examples having noise:",count)

departure_features=np.delete(departure_features,masked_index,axis=0)
label_departure_delay=np.delete(label_departure_delay,masked_index,axis=0)
print ("for tail number",departure_features.shape,label_departure_delay.shape)






# In[19]:

#for using random forest strings should be converted to integers by using  the library of label encoder
from sklearn import preprocessing
le0 = preprocessing.LabelEncoder()
le0.fit(departure_features[:,1])

departure_features[:,1]= le0.transform(departure_features[:,1])

le2 = preprocessing.LabelEncoder()
le2.fit(departure_features[:,2])

departure_features[:,2]= le2.transform(departure_features[:,2])

le3 = preprocessing.LabelEncoder()
le3.fit(departure_features[:,6])

departure_features[:,6]= le3.transform(departure_features[:,6])



print(departure_features[0:10,:])


# In[20]:

# unique1,count1=np.unique(label_departure_delay[:],return_counts=True)
# print(unique1,count1)






# In[21]:

from sklearn.preprocessing import PolynomialFeatures
# poly=PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
# departure_features=poly.fit_transform(departure_features)
    


# In[31]:


# data1 = pd.DataFrame(departure_features)
# x1 = np.array(data1[0].dropna())
# x2 = np.array(data1[1].dropna())

# x1=departure_features[:,0]
# x2=departure_features[:,1]



# x3= x1*x2
# x4= x3*x1*x2
# x5=x3*x2*x1


# X = np.vstack((x1,x2,x3)).T
# X.shape
# departure_features=preprocessing.minmax_scale(X, feature_range=(-1, 1))





#splitting data into tranning and testing test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(departure_features,label_departure_delay,test_size=0.3)#shuffle default true

Xtrain = Xtrain[:50000]
Ytrain = Ytrain[:50000]
Xtest = Xtest[:50000]
Ytest = Ytest[:50000]

# Ytrain = Ytrain.ravel()
# Ytest = Ytest.ravel()

print (" Training Data Set Dimensions=", Xtrain.shape,"Training True Class labels dimensions", Ytrain.shape)   
print (" Test Data Set Dimensions=", Xtest.shape, "Test True Class labels dimensions", Ytest.shape)  
#print(X.shape)

# masked=pd.isnull(departure_features)
#np.any(np.isnan(departure_features[:,:]))


# In[39]:

scaler = preprocessing.MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.fit_transform(Xtest)
Ytrain = scaler.fit_transform(Ytrain)
Ytest = scaler.fit_transform(Ytest)

# print(Xtrain[0:3])

# print("____________________________________________________")

# print(Xtest[0:3])
# print("____________________________________________________")

# print(Ytrain[0:3])
# print("____________________________________________________")

# print(Ytest[0:3])
# print("____________________________________________________")

Ytrain = Ytrain.ravel()
Ytest = Ytest.ravel()


# In[40]:

# count=0
# masked_index=[]
# for i in range(len(masked)):
#     for k in range(masked.shape[1]):
#         if(masked[i,k]==True):

#             print("df:i= ",i," k=","value=",masked[i,k])
#             count+=1
#             masked_index.append(i)
#             #print ("row",i)
#                 #print(X[i],Y[i])
# print ("length of masked=",len(masked_index))
# print("number of examples having noise:",count)

# departure_features=np.delete(departure_features,masked_index,axis=0)
# label_departure_delay=np.delete(label_departure_delay,masked_index,axis=0)
# print ("for 3",departure_features.shape,label_departure_delay.shape)


# In[55]:

tuned_parameters = [{'C': [1.0,0.6,0.4,0.2], 'epsilon': [0.1],
                     'kernel': ['linear','poly','rbf','sigmoid','precomputed'],'degree':[1,2,3],'gamma':['auto'],
                     'coef0':[0.0],'shrinking':[True,False], 'tol':[1e-07,1e-05,1e-03,1e-01,1e-09,1e-12,1e-15], 
                     'cache_size':[200,400,800],'max_iter':[-1]}]


# In[ ]:

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


print("Tuning hyper-parameters for accuracy")
print()
# start_time = datetime.now()

clf = GridSearchCV(SVR(), tuned_parameters, cv=3,
                   scoring='r2',n_jobs=-1,return_train_score=False)
clf.fit(Xtrain, Ytrain)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print(mean, std * 2, params)
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = Ytest, clf.predict(Xtest)
print("report->\n",classification_report(y_true, y_pred))
print()
# end_time = datetime.now()
# print('Duration: {}'.format(end_time - start_time))


# In[ ]:

Xtrain.shape


# In[ ]:

#This is the one i'm editing

