
import numpy as np
import pandas as pd
import datetime as dtt
import pickle as pk
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
import yfinance as yf
import nasdaqdatalink as ndl
import fredapi as fa
import statsmodels.stats.outliers_influence as sm

class const():
    @staticmethod
    def monthsInYear():
        return 12
    @staticmethod
    def fredAPIKey():
        return 'e3c0a8d412820723ea96a80c0abfeff0'
    @staticmethod
    def FREDTickers():
        return ['BUSLOANS','CPIAUCSL','UNRATE','PAYEMS','M1SL','M2SL','INDPRO','PSAVERT','UMCSENT','DFF','JHDUSRGDPBR','GDP']
    @staticmethod
    def YAHOOTickers():
        return ["^SPX","^DJI","BTC-USD"]
    @staticmethod
    def YAHOOStartDate():
        return '1927-01-01'
    @staticmethod
    def dtOutCols():
        return ['CI/GDPCHANGE','CPI','UNRATE','LRECM','PRCHANGE','M2CHANGE','INDPROCHANGE','PSAVERT','DFF','SPXDD','UMCSENT','RECIND']
    @staticmethod
    def regressorLags():
        return [3,3,3,3,3,3,3,3,3,3,3,3,0] #[6,2,6,6,6,4,2,6,6,5,3,0,0] # #[6,6,6,6,6,6,6,6,6,6,6,0,0]
    @staticmethod
    def NASDAQTickers():
        return ['BCHAIN/MKPRU','BCHAIN/TRVOU','BCHAIN/MWNUS','BCHAIN/MIREV','BCHAIN/HRATE','BCHAIN/CPTRA','BCHAIN/CPTRV','BCHAIN/ETRVU','BCHAIN/ETRAV','BCHAIN/NADDU','BCHAIN/NTREP',
                'BCHAIN/NTRAN','BCHAIN/TRFUS','BCHAIN/TRFEE','BCHAIN/MKTCP','BCHAIN/TOTBC','BCHAIN/BCDDE','BCHAIN/BCDDY','BCHAIN/MIOPM']
    @staticmethod
    def NASDAQColumns():
        return ['BTCPrice','BTCExchTrVol','BTCMyWallUsers','BTCMinRev','BTCHR','BTCCostPerTx','BTCCost%TxVol','BTCEstTxVolUSD','BTCEstTxVol','NoUnqBTCAddrs','BTCNoTxExclPopAddrs',
                'BTCNoTxs','BTCTotTxFeesUSD','BTCTotTxFees','BTCMktCap','TotBTCs','BTCDaysDstyd','BTCDaysDstyd1y','BTCMinOpMgn']
    @staticmethod
    def btcHalvingDates():
        return pd.to_datetime([dtt.datetime(2009,1,2,0,0,0),dtt.datetime(2012,11,28,0,0,0),
                                dtt.datetime(2016,7,9,0,0,0),dtt.datetime(2020,5,11,0,0,0),dtt.datetime(2024,4,26,11,59,22),dtt.datetime(2028,1,1,0,0,0)])
    @staticmethod
    def btcDailyRewards():
        return [50,25,12.5,6.25,3.125,1.5625]
    @staticmethod
    def C():
        return -1.84
    @staticmethod
    def expt():
        return 3.36
    @staticmethod
    def SNAdj():
        return 1e6
    @staticmethod
    def SNThreshold():
        return 10e6
    @staticmethod
    def NASDAQLags():
        return [0,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

def pull_fred_data():

    fred = fa.Fred(api_key=const.fredAPIKey())
    dt = pd.DataFrame()

    for i,p in enumerate(const.FREDTickers()):
        print(i,p)
        dt = pd.concat([dt,fred.get_series(p)],axis=1)

    dt.columns = const.FREDTickers()
    dt.index = pd.to_datetime(dt.index)

#dt.to_csv('cc.csv')
    return dt.sort_index(axis=0)

def pull_yahoo_data():

    return yf.download(const.YAHOOTickers(),start=const.YAHOOStartDate(),end=None)['Adj Close'].sort_index(axis=0)

def pull_nasdaq_data():

    ndl.read_key('.nasdaq/data_link_apikey')
    dt = ndl.get(const.NASDAQTickers())
    dt.columns = const.NASDAQColumns()
    dt.index.name = 'Date'

    return dt

def pickle_save(dt,saveStr):
# Purpose: Stores pandas dataframe as a pickled .p file

    pk.dump(dt, open('Pickled/'+saveStr, 'wb'))
    print('Pickled file ...')

def pickle_load(loadStr):
# Purpose: Loads object from a pickled .p file

    dt = pd.DataFrame()

    for l in loadStr:
        print('Loaded pickled file from: %s...' %l)
        dt = pd.concat([dt,pk.load(open('Pickled/'+l, 'rb'))],axis=0)

    return dt

def create_S2F_ts(dtNASDAQ):

    dt = pd.DataFrame(data=const.btcDailyRewards(),index=const.btcHalvingDates(),columns=['BTCDRew']).resample('D').mean().ffill() * 6 * 24
    dt = pd.merge(dt,dtNASDAQ,left_index=True,right_index=True,how='left')
    dt.index.name = 'Date'

    # create adjusted columns to account for 1MM BTC released by SN
    arrAdj = dt['TotBTCs'].copy().values
    nanIndx = np.where(np.isnan(arrAdj))
    arrAdj[nanIndx] =  arrAdj[np.min(nanIndx)-1] + np.cumsum(dt['BTCDRew'].iloc[nanIndx]).values
    dt['TotBTCs'] = arrAdj
    redIndx = np.where(arrAdj >= const.SNThreshold())
    arrAdj[redIndx] = arrAdj[redIndx] - const.SNAdj()
    dt['TotBTCsAdj'] = arrAdj

    dt['StoF'] = dt['TotBTCs'] / (365 * dt['BTCDRew'])
    dt['StoFAdj'] = dt['TotBTCsAdj'] / (365 * dt['BTCDRew'])
    dt['ModPrice'] = np.exp(const.C()) * np.power(dt['StoF'],const.expt())
    dt['ModPriceAdj'] = np.exp(const.C()) * np.power(dt['StoFAdj'],const.expt())

    dt = dt[['BTCDRew', 'TotBTCs', 'TotBTCsAdj', 'StoF', 'StoFAdj','BTCPrice','ModPrice', 'ModPriceAdj']]
    dt.to_csv('sf.csv')

    return dt

def format_data(dtFRED,dtYAHOO):

    # clean data so that it matches article
    dt = dtFRED.drop(dtFRED.index[np.where(dtFRED.index<dtt.datetime(1970,6,1,0,0,0))],axis=0)
    dt.fillna(method='ffill',axis=0,inplace=True)
    dt = dt.resample('M',axis=0).mean()
    dt = pd.merge(dt,pd.DataFrame(data=dtYAHOO, index=pd.date_range(dtYAHOO.index.min(), dtYAHOO.index.max(), freq='D'),columns=dtYAHOO.columns).fillna(method='ffill', axis=0),
                  how='left',left_index=True,right_index=True)
    dt.index.name = 'Date'
    #dt.to_csv('dt.csv')

    return dt

def create_regressors(dtFRED):

    dtOut = pd.DataFrame(data=0,index=dtFRED.index,columns=const.dtOutCols())

    # calculate CI Loans / GDP MoM change
    dtOut['CI/GDPCHANGE'] = (dtFRED['BUSLOANS'] / dtFRED['GDP']).pct_change()
    # calculate MoM changes
    dtOut[['CPI','M2CHANGE','INDPROCHANGE','PRCHANGE','BTC-USD']] = dtFRED[['CPIAUCSL','M2SL','INDPRO','PAYEMS','BTC-USD']].pct_change()
    # include levels
    dtOut[['UNRATE','PSAVERT','DFF','UMCSENT','RECIND']] = dtFRED[['UNRATE','PSAVERT','DFF','UMCSENT','JHDUSRGDPBR']]

    # calculate months since last recession (ends up as 1 within recessions)
    recIndx = dtFRED.index[np.where(dtFRED['JHDUSRGDPBR'].values == 1)]
    dtOut['LREC'] = recIndx[np.searchsorted(recIndx,dtFRED.index)-1]
    dtOut['LRECM'] = dtOut.index.to_period('M').astype(int) - dtOut['LREC'].dt.to_period('M').astype(int)
    dtOut.loc[dtOut.index[np.where(dtOut['LRECM'].values<0)],'LRECM'] = np.nan
    dtOut.drop('LREC', axis=1, inplace=True)

#dtOut.to_csv('dtOut.csv')
#dtFRED.to_csv('dtFRED.csv')

    # calculate 1 - SPX price / last 12m max SPX price
    arrSPX = np.full((dtOut.shape[0]),np.nan)
    for i in range(12,dtOut.shape[0]):
        arrSPX[i] = 1 - dtFRED['^SPX'].iloc[i] / np.nanmax(dtFRED['^SPX'].iloc[i-12:i])
        #print(i,dtFRED['^SPX'].iloc[i],np.nanmax(dtFRED['^SPX'].iloc[i-12:i]))
    dtOut['SPXDD'] = arrSPX

    return dtOut.dropna()

def lag_data(dt,lags):

    for i,c in enumerate(dt.columns):
        print(c)
        dt[c] = dt[c].shift(periods=lags[i])

    dt.dropna(inplace=True)

    return dt



def create_GB_classifier_model(dtRegr,dtResp):

    modelGB = sklearn.ensemble.GradientBoostingClassifier()
    Xtrain,Xtest,ytrain,ytest = sklearn.model_selection.train_test_split(dtRegr,dtResp,test_size=0.33,random_state=42)

    return modelGB.fit(Xtrain,ytrain),Xtest,ytest

def create_LR_classifier_model(dtRegr,dtResp):

    modelLR = sklearn.linear_model.LogisticRegression(C=1000,penalty='l1',solver='saga',max_iter=10000)
    Xtrain,Xtest,ytrain,ytest = sklearn.model_selection.train_test_split(dtRegr,dtResp,test_size=0.33,random_state=42)

    return modelLR.fit(Xtrain,ytrain),Xtest,ytest

def create_SGD_regression_model(dtRegr,dtResp):

    modelSGD = sklearn.linear_model.SGDRegressor()
    Xtrain,Xtest,ytrain,ytest = sklearn.model_selection.train_test_split(dtRegr,dtResp,test_size=0.33,random_state=42)

    return modelSGD.fit(Xtrain,ytrain),Xtest,ytest

def create_Lasso_regression_model(dtRegr,dtResp):

    modelLasso = sklearn.linear_model.LassoLARS()
    Xtrain,Xtest,ytrain,ytest = sklearn.model_selection.train_test_split(dtRegr,dtResp,test_size=0.33,random_state=42)

    return modelLasso.fit(Xtrain,ytrain),Xtest,ytest

def analyze_GBmodel(model,Xtest,ytest):

#model = modelGB
#Xtest = XtGB
#ytest = ytGB

    print('\nModel overall accuracy is: %.2f for GB model ...' \
          % (model.score(Xtest, ytest)))

    print('\nModel roc auc score is: %.2f for GB model ...' \
          % (sklearn.metrics.roc_auc_score(ytest,modelGB.predict(Xtest))))

    # print precision and recall scores
    ymod = model.predict(Xtest)
    print('Model precision and recall scores are: %.4f and %.4f for GB model ...' \
          % (sklearn.metrics.precision_score(ytest, ymod, average='macro'),
             sklearn.metrics.recall_score(ytest, ymod, average='macro')))

    fiGB = pd.DataFrame({'Feature': modelGB.feature_names_in_, 'Importance': np.abs(modelGB.feature_importances_)}).sort_values('Importance', ascending=False)
    return fiGB

def analyze_LRmodel(model, Xtest, ytest):

#model = modelLR
#Xtest = XtLR
#ytest = ytLR

    print('\nModel overall accuracy is: %.2f for LR model ...' \
          % (model.score(Xtest, ytest)))

    print('\nModel roc auc score is: %.2f for LR model ...' \
          % (sklearn.metrics.roc_auc_score(ytest, modelLR.predict(Xtest))))

# print precision and recall scores
    ymod = model.predict(Xtest)
    print('Model precision and recall scores are: %.4f and %.4f for LR model ...' \
          % (sklearn.metrics.precision_score(ytest, ymod, average='macro'),
             sklearn.metrics.recall_score(ytest, ymod, average='macro')))

    fiLR = pd.DataFrame({'Feature': model.feature_names_in_, 'Importance': np.abs(model.coef_[0])}).sort_values('Importance', ascending=False)

    return fiLR

def calculate_vif_table(X):

    dtVif = pd.DataFrame()
    dtVif['feature'] = X.columns
    dtVif["VIF"] = [sm.variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(dtVif)

def calculate_correlation_matrix(dt,plotFlag):

    if (plotFlag == True):
        plt.matshow(dt.corr(method='spearman'))
        plt.show()

    #dt.corr(method='pearson').to_csv('corrp.csv')
    #dt.corr(method='spearman').to_csv('corrs.csv')
    print(dt.corr(method='spearman'))
    #return dt.corr(method='spearman')


#plt.rcParams.update({'font.size': 8})
#f, axArr = plt.subplots(2, 1)
#fiGB.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
#fiLR.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

#oo = pd.concat([Xtest,ytest,dtPGB,dtPLR],axis=1)

def main(argv=sys.argv):

dtFRED = pull_fred_data()
dtYAHOO = pull_yahoo_data()
dtNASDAQ = pull_nasdaq_data()

dtNASDAQLag = lag_data(dtNASDAQ.copy(deep=True).resample('M').last(),const.NASDAQLags())
dtNASDAQLag.drop(dtNASDAQLag.index[np.where(dtNASDAQLag.index < dtt.datetime(2016,1,1,0,0,0))],axis=0,inplace=True)
calculate_vif_table(dtNASDAQLag.pct_change().iloc[:,2:].dropna())
calculate_correlation_matrix(dtNASDAQLag.pct_change(),False)
dtS2F = create_S2F_ts(dtNASDAQ)

dt = format_data(dtFRED,dtYAHOO)
dt = create_regressors(dt)
dtLag = lag_data(dt.copy(deep=True),const.regressorLags())
#dtLag.drop(['UNRATE','PSAVERT','UMCSENT','RECIND'],axis=1,inplace=True)
#dt.drop(['UNRATE','PSAVERT','UMCSENT','RECIND'],axis=1,inplace=True)

calculate_vif_table(dtLag.iloc[:,:-1])
calculate_correlation_matrix(dtLag,False)

dtB = dt[['CPI','DFF','SPXDD','RECIND','BTC-USD']]
dtBLag = dtLag[['CPI','DFF','SPXDD','RECIND','BTC-USD']]
#modelGB,XtGB,ytGB = create_GBmodel(dtBLag[dtBLag.columns[:-1]],dtBLag['BTC-USD'])
#modelLR,XtLR,ytLR = create_LRmodel(dtBLag[dtBLag.columns[:-1]],dtBLag['BTC-USD'])

modelGB,XtGB,ytGB = create_SGD_regression+model(dtBLag[dtBLag.columns[:-1]],dtBLag['BTC-USD'])

fiGB = analyze_GBmodel(modelGB,XtGB,ytGB)
fiLR = analyze_LRmodel(modelLR,XtLR,ytLR)

dtP = pd.DataFrame(data=np.transpose(np.squeeze(np.array([[modelGB.predict_proba(dtB[dtB.columns[:-1]])[:,1]],[modelLR.predict_proba(dtB[dtB.columns[:-1]])[:,1]]]))),
                   index=dtB.index,columns=['Pr GB','Pr LR'])
pd.concat([dtB,dtP],axis=1).to_csv('out.csv')


if __name__ == "__main__":
    sys.exit(main())
