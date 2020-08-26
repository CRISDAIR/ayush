import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from datetime import date,timedelta
import cx_Oracle as dbc
#import csv
import getpass
import warnings
warnings.filterwarnings("ignore")
def TOTAL_CLOSE_DEMAND(df2):
    CLOSE=[]
    TOTAL_DEMAND=[]
    MET=df2['DEMAND_MET'].values
    DEMAND=df2['DEMAND'].values
    for i in range(len(df2)):
        if(i==0):
            CLOSE.append(DEMAND[i]-MET[i])
        else:
            CLOSE.append(DEMAND[i]+CLOSE[i-1]-MET[i])
    df2['DEMAND_CLOSE']=pd.DataFrame(CLOSE)
    for i in range(len(df2)):
        if(i==0):
            TOTAL_DEMAND.append(DEMAND[i])
        else:
            TOTAL_DEMAND.append(DEMAND[i]+CLOSE[i-1])
    df2.insert(1,'TOTAL_DEMAND',pd.DataFrame(TOTAL_DEMAND))
    return df2
def MAX_MIN_DEMAND(df2):
    TOTAL_DEMAND=df2['TOTAL_DEMAND'].values
    MAX=[]
    MIN=[]
    for i in range(len(df2)):
        sub=[]
        if(i<5):
           sub=TOTAL_DEMAND[0:i+1]
           y=max(sub)
           MAX.append(y)
        else:
            sub=TOTAL_DEMAND[i-5:i]
            y=max(sub)
            MAX.append(y)
    for i in range(len(df2)):
        sub=[]
        if(i<5):
           sub=TOTAL_DEMAND[0:i+1]
           y=min(sub)
           MIN.append(y)
        else:
            sub=TOTAL_DEMAND[i-5:i]
            y=min(sub)
            MIN.append(y)
    df2['MAX_DEMAND']=pd.DataFrame(MAX)
    df2['MIN_DEMAND']=pd.DataFrame(MIN)
    return df2
def Model(df2):
    last_day=df2.tail(5)
    Z=list(last_day.index.values)
    train=df2[df2['DATE']<last_day.at[Z[4], 'DATE']].copy()
    test=df2[df2['DATE']>=last_day.at[Z[0], 'DATE']].copy()
    train=train.drop(['DATE','DEMAND','DEMAND_MET'],axis=1)
    test=test.drop(['DATE','DEMAND','DEMAND_MET'],axis=1)
    train_1=train
    start_4_days=test.head(4)
    train_1=train_1.append(start_4_days)
    X_train=[]
    Y_train=[]
    train=np.array(train)
    for i in range(5,train.shape[0]):
        X_train.append(train[i-5:i])
    for i in range(5,len(train_1)-4):
        Y_train.append(train_1['TOTAL_DEMAND'][i:i+5])
    X_train_1=np.array(X_train)
    Y_train_1=np.array(Y_train)
    model = Sequential()
    model.add(LSTM(200, activation='relu',return_sequences=True,input_shape=(X_train_1.shape[1], 4)))
    model.add(LSTM(250, activation='relu',return_sequences=True))
    model.add(LSTM(300, activation='relu',return_sequences=True))
    model.add(LSTM(350, activation='relu',return_sequences=True))
    model.add(LSTM(400, activation='relu'))
    model.add(Dense(units=5))
    model.summary()
    model.compile(optimizer="adam",loss="mean_squared_error")
    model.fit(X_train_1,Y_train_1,epochs=200,batch_size=6)
    return (model)
def train_station(df_SID,x,z,STTNFROM):
    for j in range(len(STTNFROM)):
        df_DATA=df_SID.loc[df_SID['STTNFROM'].isin([STTNFROM[j]])]
        df_DATA=df_DATA.sort_values(by=['DMNDDATE'])
        df2=df_DATA.groupby('DMNDDATE')['RANFRWH'].sum()
        idx = pd.date_range(x,z)
        df2.index = pd.DatetimeIndex(df2.index)
        df2 = df2.reindex(idx, fill_value=0)
        df_DATA=df_DATA.sort_values(by=['METDATE'])
        df3=df_DATA.groupby('METDATE')['RANFRWH'].sum()
        idx = pd.date_range(x,z)
        df3.index = pd.DatetimeIndex(df3.index)
        df3 = df3.reindex(idx, fill_value=0)
        df3.columns=['DATE','DEMAND_MET']
        #df3.rename(columns={'RANFRWH':'DEMAND_MET'},inplace=True)
        df2=pd.concat([df2,df3],axis=1)
        df2=df2.reset_index()
        df2.columns=['DATE','DEMAND','DEMAND_MET']
        #df2.rename(columns={df2.columns[0]:'DATE',df2.columns[1]:'DEMAND',df2.columns[2]:'DEMAND_MET'},inplace=True)
        df2=TOTAL_CLOSE_DEMAND(df2)
        df2=MAX_MIN_DEMAND(df2)
        y=Model(df2)
        y.save(STTNFROM[j])
def train(df,x,z):
    new = df["RADDMNDTIME"].str.split(" ", n = 1, expand = True) 
    df['DMNDDATE']=new[0]
    new_1 = df["RADMETWITHDATE"].str.split(" ", n = 1, expand = True)
    df['METDATE']=new_1[0]
    df_SID=df.loc[df['RAVGRUPRAKECMDT'].isin(["CEMT"])]
    STTNFROM=df_SID['STTNFROM'].unique()
    train_station(df_SID,x,z,STTNFROM)
        
#userpwd=getpass.getpass(prompt="Enter the Password")
#pool1=dbc.SessionPool("kunalmalhan", userpwd,"10.30.2.202:1521/FOIS", min=2, max=5, increment=1)
#con1=pool1.acquire()
#cur1=con1.cursor()
#r = cur1.execute("SELECT	DM.RAVDMNDID, DM.RAVRAKETYPE, DM.RAVCNSR, DM.RAVCNSG, "
# " 		DM.RADDMNDTIME, DM.RADEXPDLDNGDATE, DM.RADMETWITHDATE, DM.RACDMNDSTTS, "
#+ " 		MF.MAVSTTNWRKGZONECODE||'/'||MF.MAVDVSNCODE||'/'||DECODE(MF.MAVSTTNALFACODE, DM.RAVSRVGSTTN, '', DM.RAVSRVGSTTN||'/')||MF.MAVSTTNALFACODE  STTNFROM, "
#+ " 		DM.RAVGRUPRAKECMDT, "
#+ " 		DM.RANFRWH, "
#+ " 		DT.RANSQNCNUMB, "
#+ " 		MT.MAVSTTNWRKGZONECODE||'/'||MT.MAVDVSNCODE||'/'||MT.MAVSTTNALFACODE STTNTO"
#+ " 		FROM	"
#+ " 		("
#+ " 			SELECT * FROM RMS3T.REM_DMND"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCH3T.REM_DMND"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCHBKP.REM_DMND@FOIS_TOREAD_DATA1920"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCHBKP.REM_DMND@FOIS_TOREAD_DATA2021"
#+ " 		) DM, "
#+ " 		("
#+ " 			SELECT * FROM RMS3T.RED_DMND"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCH3T.RED_DMND"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCHBKP.RED_DMND@FOIS_TOREAD_DATA1920"
#+ " 			UNION ALL"
#+ " 			SELECT * FROM RMSARCHBKP.RED_DMND@FOIS_TOREAD_DATA2021"
#+ " 		)  DT, MEMSTTN MF, MEMSTTN MT"
#+ " 		WHERE	DM.RAVDMNDID	= DT.RAVDMNDID"
#+ " 		AND		DT.RAVSTTNTO	= MT.MAVSTTNALFACODE"
#+ " 		AND		TRUNC(NVL(DM.RADEXPYTIMEFINL, SYSDATE)) >= TRUNC(SYSDATE)"
#+ " 		AND		DM.RADDMNDTIME	>=TO_DATE('01-01-2019','DD-MM-YYYY')"
#+ " 		AND		DM.RADDMNDTIME	< TO_DATE('31-07-2020','DD-MM-YYYY') + 1"
#+ " 		AND		DM.RANGAUG		= 1"
#+ " ORDER BY DM.RAVDMNDID, DT.RANSQNCNUMB")
#df=pd.DataFrame(r.fetchall())
#columns=['RAVDMNDID' 'RAVRAKETYPE' 'RAVCNSR' 'RAVCNSG' 'RADDMNDTIME'
 #'RADEXPDLDNGDATE' 'RADMETWITHDATE' 'RACDMNDSTTS' 'STTNFROM'
 #'RAVGRUPRAKECMDT' 'RANFRWH' 'RANSQNCNUMB' 'STTNTO']
#df.columns=columns
#today=date.today()
#d=today.strftime("%Y/%m/%d") # Date of prediction
#d1 = date.today() - timedelta(days=1) #Date previous day of predcition/upto which data required for prediction
#d2 = d1 - timedelta(days=9) #date from which we have to take data
#d1=d1.strftime("%Y/%m/%d")
#d2=d2.strftime("%Y/%m/%d")
d2='2019/01/01'
d1='2019/12/31'
#df['RADDMNDTIME']=df['RADDMNDTIME'].astype(str)
#df['RADMETWITHTIME']=df['RADMETWITHTIME'].astype(str)
df=pd.read_csv('DEMAND_DATA.csv')
x=train(df,d2,d1)
