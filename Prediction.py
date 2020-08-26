import pandas as pd
import numpy as np
from datetime import date,timedelta
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Dropout
from sklearn.metrics import r2_score
import cx_Oracle as dbc
import csv
import getpass
from datetime import date
import warnings
warnings.filterwarnings("ignore")
def ACCURACY(Demand_actual,Demand_predicted):
    x=Demand_actual-Demand_predicted
    x=abs(x)
    percentage_error=(x/Demand_actual)*100
    return (percentage_error)        
def TOTAL_CLOSE_DEMAND(df2):
    CLOSE=[]
    TOTAL_DEMAND=[]
    MET=df2['DEMAND_MET'].values
    DEMAND=df2['DEMAND'].values
    for i in range(len(df2)):
        if(i==0 and DEMAND[i]>=MET[i]):
            CLOSE.append(DEMAND[i]-MET[i])
        elif(i==0 and DEMAND[i]<MET[i]):
            CLOSE.append(0)
        elif(i!=0 and(DEMAND[i]+CLOSE[i-1])<MET[i]):
            CLOSE.append(0)
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
def Prediction(df2,r): # function to make prediction
    reconstructed_model = keras.models.load_model(r)
    df2=df2.drop(['DATE','DEMAND','DEMAND_MET'],axis=1)
    df3=df2.tail(5)
    test=np.array(df3)
    X_test=[]
    X_test.append(test[0:5])
    X_test=np.array(X_test)
    w=reconstructed_model.predict((X_test))
    return (w)
def Predict(df,d2,d1,d,cur1):
    # function to make data set for prediction and pass data frame to function prediction required to make prediction
    #Output=pd.read_csv('Output.csv',delim_whitespace=True)
    col=['DATE','STTNFROM','t-1','RMSE','t','t+1','t+2','t+3','t+4']
    C=[]
    new = df["RADDMNDTIME"].str.split(" ", n = 1, expand = True) 
    df['DMNDDATE']=new[0]
    new_1 = df["RADMETWITHDATE"].str.split(" ", n = 1, expand = True)
    df['METDATE']=new_1[0]
    df_SID=df.loc[df['RAVGRUPRAKECMDT'].isin(["CEMT"])]
    df_out=df_SID
    STTNFROM=df_SID['STTNFROM'].unique()
    DATE=[d]*(5)
    t_prev=[]
    RMSE=[]
    t_curr=[]
    for j in range(5):
        df_DATA=df_SID.loc[df_SID['STTNFROM'].isin([STTNFROM[j]])]
        df_DATA=df_DATA.sort_values(by=['DMNDDATE'])
        df2=df_DATA.groupby('DMNDDATE')['RANFRWH'].sum()
        idx = pd.date_range(d2,d1)
        df2.index = pd.DatetimeIndex(df2.index)
        df2 = df2.reindex(idx, fill_value=0)
        df_DATA=df_DATA.sort_values(by=['METDATE'])
        df3=df_DATA.groupby('METDATE')['RANFRWH'].sum()
        idx = pd.date_range(d2,d1)
        df3.index = pd.DatetimeIndex(df3.index)
        df3 = df3.reindex(idx, fill_value=0)
        df3.columns=['DATE','DEMAND_MET']
       # df3.rename(columns={'RANFRWH':'DEMAND_MET'},inplace=True)
        df2=pd.concat([df2,df3],axis=1)
        df2=df2.reset_index()
        df2.columns=['DATE','DEMAND','DEMAND_MET']
        #df2.rename(columns={df2.columns[0]:'DATE',df2.columns[1]:'DEMAND',df2.columns[2]:'DEMAND_MET'},inplace=True)
        df2=TOTAL_CLOSE_DEMAND(df2)
        t_curr.append(df2['TOTAL_DEMAND'][9])
        df2=MAX_MIN_DEMAND(df2)
        y=Prediction(df2,STTNFROM[j])
        y=y.flatten()
        C.append(y.tolist())
    t=cur1.execute("SELECT * FROM DEMAND_PRIDICTION WHERE DATE==(d1,'DD-MM-YYYY')")
    out=pd.DataFrame(t.fetchall())
    if(not(out.empty)):
        out.columns=col
        for i in range(len(STTNFROM)):
            df_out=out.loc[out['STTNFROM'].isin([STTNFROM[i]])]
            B=df_out['t'].values
            t_prev.append(B[0])
            m=ACCURACY(t_curr[i],B[0])
            RMSE.append(m)
    else:
        t_prev=[0]*5
        RMSE=[0]*5 
    Output=pd.DataFrame(C,columns=['t','t+1','t+2','t+3','t+4'])
    Output.insert(0,'RMSE',pd.DataFrame(RMSE))
    Output.insert(0,'t_prev',pd.DataFrame(t_prev))
    #Output.insert(0,'STTNFROM',pd.DataFrame(STTNFROM))
    Output.insert(0,'DATE',pd.DataFrame(DATE))
    sql='insert into DEMAND_PRIDICTION values(:1,:2,:3,:4,:5,:6)'
    df_list = Output.values.tolist()
    n = 0
    for i in Output.iterrows():
        cur1.execute(sql,df_list[n])
        n += 1
   # (Output.to_csv("Output.csv",index=False))
today=date.today()
d=today.strftime("%Y-%m-%d") # Date of prediction
d1 = date.today() - timedelta(days=1) #Date previous day of predcition/upto which data required for prediction
d2 = d1 - timedelta(days=9) #date from which we have to take data
d1=d1.strftime("%Y-%m-%d")
d2=d2.strftime("%Y-%m-%d")
d2='2020-08-01'
d1='2020-12-10'
pool1=dbc.SessionPool("kunalmalhan","Kunal#678","10.30.2.202:1521/FOIS", min=2, max=5, increment=1)
con1=pool1.acquire()
cur1=con1.cursor()
r=cur1.execute(" SELECT	DM.RAVDMNDID,DM.RAVRAKETYPE, DM.RAVCNSR, DM.RAVCNSG, "
+ " 		DM.RADDMNDTIME, DM.RADEXPDLDNGDATE, DM.RADMETWITHDATE, DM.RACDMNDSTTS,"
+ " 		MF.MAVSTTNWRKGZONECODE||'/'||MF.MAVDVSNCODE||'/'||DECODE(MF.MAVSTTNALFACODE, DM.RAVSRVGSTTN, '', DM.RAVSRVGSTTN||'/')||MF.MAVSTTNALFACODE  STTNFROM, "
+ " 		DM.RAVGRUPRAKECMDT, "
+ " 		DM.RANFRWH, "
+ " 		DT.RANSQNCNUMB, "
+ " 		MT.MAVSTTNWRKGZONECODE||'/'||MT.MAVDVSNCODE||'/'||MT.MAVSTTNALFACODE STTNTO"
+ " 		FROM	REM_DMND DM, RED_DMND DT, MEMSTTN MF, MEMSTTN MT"
+ " 		WHERE	DM.RAVDMNDID	= DT.RAVDMNDID"
+ " 		AND		DM.RAVSTTNFROM	= MF.MAVSTTNALFACODE"
+ " 		AND		DT.RAVSTTNTO	= MT.MAVSTTNALFACODE"
+ " 		AND		TRUNC(NVL(DM.RADEXPYTIMEFINL, SYSDATE)) >= TRUNC(SYSDATE)"
+ " 		AND		DM.RADDMNDTIME	>=TO_DATE('04-08-2020','DD-MM-YYYY')"
+ " 		AND		DM.RADDMNDTIME	< TO_DATE('09-08-2020','DD-MM-YYYY') + 1"
+ " 		AND		DM.RANGAUG		= 1"
+ " ORDER BY DM.RAVDMNDID, DT.RANSQNCNUMB")
df=pd.DataFrame(r.fetchall())
columns=['RAVDMNDID', 'RAVRAKETYPE', 'RAVCNSR', 'RAVCNSG', 'RADDMNDTIME',
 'RADEXPDLDNGDATE', 'RADMETWITHDATE', 'RACDMNDSTTS', 'STTNFROM',
 'RAVGRUPRAKECMDT' ,'RANFRWH','RANSQNCNUMB', 'STTNTO']
df.columns=columns
df['RADDMNDTIME']=df['RADDMNDTIME'].astype(str)
df['RADMETWITHTIME']=df['RADMETWITHTIME'].astype(str)
df=pd.read_csv('DEMAND_DATA.csv')
x=Predict(df,d2,d1,d,cur1)
