import wrds
import pandas_datareader
import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from decimal import Decimal
import scipy
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

load_data = False
data_folder = '/Users/Xinlin/Desktop/Quantitative Asset Management/Problem Sets/PS3/'
wrds_id = 'yxinlin'

min_shrcd = 10
max_shrcd = 11
min_year = 1926
max_year = 2021
possible_exchcd = (1, 2, 3)

if load_data:
    conn = wrds.Connection(wrds_username=wrds_id)
    # load CRSP returns
    crsp_raw = conn.raw_sql("""
                          select a.permno, a.date, a.ret, a.shrout, a.prc,
                          b.shrcd, b.exchcd
                          from crspq.msf as a
                          left join crspq.msenames as b
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
                          and a.date between '01/01/""" +str(min_year)+ """' and '12/31/""" +str(max_year)+ """'
                          and b.exchcd in """ + str(possible_exchcd) + """
                          """)
    # load CRSP delisting returns
    dlret_raw = conn.raw_sql("""
                            select permno, dlstdt, dlret, dlstcd
                            from crspq.msedelist
                            """)
    conn.close()
    
    # save csv
    crsp_raw.to_csv('crsp_raw.csv')
    dlret_raw.to_csv('dlret_raw.csv')

else:
    crsp_raw = pd.read_csv('crsp_raw.csv')
    dlret_raw = pd.read_csv('dlret_raw.csv')                      
    
## data cleaning
# change variables
crsp_raw['permno'] = crsp_raw['permno'].astype(int)
crsp_raw['date'] = pd.to_datetime(crsp_raw['date'], format='%Y-%m-%d', errors='ignore')
crsp_raw['date'] = pd.DataFrame(crsp_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
crsp_raw = crsp_raw.sort_values(by=['date', 'permno'])

dlret_raw.permno = dlret_raw.permno.astype(int)
dlret_raw['date'] = pd.to_datetime(dlret_raw['dlstdt'], format='%Y-%m-%d', errors='ignore')
dlret_raw['date'] = pd.DataFrame(dlret_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
dlret_raw = dlret_raw.sort_values(by=['date', 'permno'])

# merge holding period return data with delisted return data
crsp = crsp_raw.merge(dlret_raw, how='outer', on=['date', 'permno'])

# compute adjusted returns
crsp['ret'] = np.where(crsp['ret'].notna() & crsp['dlret'].notna(), 
                       (1+crsp['ret'])*(1+crsp['dlret'])-1, crsp['ret'])
crsp['ret'] = np.where(crsp['ret'].isna() & crsp['dlret'].notna(),
                       crsp['dlret'], crsp['ret'])
crsp = crsp[crsp['ret'].notna()]

# final input
crsp = crsp[['date','permno','exchcd','prc','shrout','ret']].\
              sort_values(by=['date','permno']).reset_index(drop=True)
crsp_stocks = crsp.copy()

#load Fama-French Data
load_data = False

if load_data:
    # F-F 3 Factors
    pd.set_option('precision', 4)
    ff3 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='1926', end='2022')
    ff3 = ff3.read()[0]/100
    ff3.columns = 'MktRF', 'SMB', 'HML', 'RF'
    ff3['Mkt'] = ff3['MktRF'] + ff3['RF']
    ff3.to_csv('ff3.csv')
    
    # F-F Momentum Factors
    ff_m = pandas_datareader.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start='1926',end='2022')
    ff_m = ff_m.read()[0]/100
    ff_m.rename(columns={'Mom':'UMD'}, inplace=True)
    ff_m.to_csv('ff_m.csv')
    
    # F-F Momentum Portfolios
    ff_p = pandas_datareader.famafrench.FamaFrenchReader('10_Portfolios_Prior_12_2',start='1926',end='2022')
    ff_p = ff_p.read()[0]/100
    ff_p.columns = 'M01','M02','M03','M04','M05','M06','M07','M08','M09','M10'
    ff_p.to_csv('ff_p.csv')

else:
    ff3 = pd.read_csv('ff3.csv')
    ff_m = pd.read_csv('ff_m.csv')
    ff_p = pd.read_csv('ff_p.csv')
    
ff3['date'] = pd.to_datetime(ff3['Date'], format='%Y-%m-%d', errors='ignore')
ff3['date'] = pd.DataFrame(ff3[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)


# compute mkt_cap
crsp['me'] = crsp['prc'].abs() * crsp['shrout']

# find lag_mkt_cap
crsp['new_date'] = crsp['date'] + dt.timedelta(days=20)
crsp['new_date'] = pd.DataFrame(crsp[['new_date']].values.astype('datetime64[ns]')) + MonthEnd(0)
lcrsp = crsp[['new_date','permno','me']].rename(columns={'new_date':'date', 'me':'lag_mkt_cap'})
crsp = crsp[['date','permno','exchcd','ret','me']]
crsp = crsp.merge(lcrsp, how='inner', on=['date','permno'])
crsp = crsp[crsp['ret'].notna() & crsp['lag_mkt_cap'].notna()]
crsp = crsp.drop_duplicates(subset=['date','permno'])

# compute past returns: cumulative log return (t-12 ~ t-2)
df = crsp.sort_values(by=['permno','date'])
df = df.loc[df['lag_mkt_cap']>0]
df['log_ret'] = np.log(1 + df['ret'].shift(1))  # log return (lag 1 month)
df_ret = df.groupby(['permno']).rolling(window=11, on='date')['log_ret'].sum().reset_index()\
         .rename(columns={'log_ret':'ret12m'}).dropna()

# find ranking returns
df = pd.merge(df, df_ret, how='left', on=['permno','date'])
df = df[df['date']>=pd.to_datetime('1927-01-01')]
df = df[df['ret12m'].notna()]
df = df.sort_values('date')
ranking_ret = df.groupby('date')['ret12m'].rank(method='max')\
              .reset_index().rename(columns={'ret12m':'ranking_ret'})

df = df.reset_index()
df = pd.merge(df, ranking_ret, how='left', on=['index']).reset_index(drop=True)
q1 = df[['date','permno','lag_mkt_cap','ret12m','ranking_ret']]


# find Daniel & Moskowitz portfolio decile of each stock
K = 10 # number of percentile
s = 'ret12m' # signal

df['DM_decile'] = np.nan
df['DM_decile'] = df.groupby('date')[s].transform(lambda x: pd.qcut(x, K, labels=np.arange(1,K+1)))
df['DM_decile'] = df['DM_decile'].astype('float')

# find Kenneth R. French's portfolio decile of each stock
def bins(df,K,s,use_cutoff_flag=False):
    df = df.copy()
    
    # Defining the momentum percentiles
    df['brkpts_flag'] = True
    if use_cutoff_flag:
        df['brkpts_flag'] = (df['exchcd'] == 1)

    def diff_brkpts(x,K):
        brkpts_flag = x['brkpts_flag']
        x = x[s]
        loc_nyse = x.notna() & brkpts_flag  
        if np.sum(loc_nyse) > 0:
            breakpoints = pd.qcut(x[loc_nyse], K, retbins=True, labels=False)[1]
            breakpoints[0] = -np.inf
            breakpoints[K] = np.inf
            y = pd.cut(x, bins=breakpoints, labels=False) + 1
        else:
            y = x + np.nan
        return y

    df['KRF_decile'] = df.groupby('date').apply(lambda x:diff_brkpts(x,K)).reset_index()[s]
    df.drop('brkpts_flag',axis=1,inplace=True)
    return df

df = bins(df,K,s,use_cutoff_flag=True)
df = df[df['date']<=pd.to_datetime('2021-12-31')]
q2 = df[['date','permno','lag_mkt_cap','ret12m','DM_decile','KRF_decile']]

# compute value-weighted returns for each decile
sum_mv_dm = df.groupby(['date', 'DM_decile'])['lag_mkt_cap'].sum().reset_index()\
         .rename(columns={'lag_mkt_cap': 'sum_mv_dm'})
sum_mv_krf = df.groupby(['date', 'KRF_decile'])['lag_mkt_cap'].sum().reset_index()\
         .rename(columns={'lag_mkt_cap': 'sum_mv_krf'})
df = pd.merge(df, sum_mv_dm, how='left', on=['date','DM_decile'])
df = pd.merge(df, sum_mv_krf, how='left', on=['date','KRF_decile'])

df['vw_DM_ret'] = (df['lag_mkt_cap']/df['sum_mv_dm']) * df['ret']
df['vw_KRF_ret'] = (df['lag_mkt_cap']/df['sum_mv_krf']) * df['ret']

DM_ret = df.groupby(['date', 'DM_decile'])['vw_DM_ret'].sum().reset_index()\
         .rename(columns={'vw_DM_ret': 'DM_ret'})
KRF_ret = df.groupby(['date', 'KRF_decile'])['vw_KRF_ret'].sum().reset_index()\
         .rename(columns={'vw_KRF_ret': 'KRF_ret', 'KRF_decile':'DM_decile'})

q3 = pd.merge(DM_ret, KRF_ret, how='left', on=['date','DM_decile']).rename(columns={'DM_decile':'decile'})
q3 = pd.merge(q3, ff3[['RF','date']], how='left', on=['date'])

q4 = q3.drop('KRF_ret', axis=1)
q4['ex_ret'] = q4['DM_ret'] - q4['RF']
q4['log_ret'] = np.log(q4['DM_ret'] + 1)

# compute annualized mean excess return
ex_ret = q4.groupby('decile')['ex_ret'].mean().reset_index()
ex_ret['annualized_ex_ret'] = ex_ret['ex_ret']*12*100

# compute annualized standard deviation of return
std = q4.groupby('decile')['ex_ret'].std().reset_index()
std['annualized_std'] = std['ex_ret']*np.sqrt(12)*100

# compute annualized Sharpe Ratio
table = pd.merge(ex_ret, std, how='left', on=['decile'])
table['sharpe_ratio'] = table['annualized_ex_ret']/table['annualized_std']

# compute skewness of log return
from scipy.stats import skew
sk = q4.groupby('decile')['log_ret'].skew().reset_index().rename(columns={'log_ret':'skewness_logret'})

table = pd.merge(table, sk, how='left', on=['decile']).drop(['ex_ret_x','ex_ret_y'], axis=1)

# compute statistics for WML portfolio
q4['WML_long'] = np.where(q4['decile']==10, 1, 0)
q4['WML_short'] = np.where(q4['decile']==1, -1, 0)
q4['WML_ret'] = q4['WML_long']*q4['DM_ret'] + q4['WML_short']*q4['DM_ret']
wml = q4.groupby('date')['WML_ret'].sum().reset_index()
wml = pd.merge(wml, ff3[['date','RF']], how='left', on=['date'])
wml['WML_log_ret'] = np.log(1 + wml['WML_ret'] + wml['RF'])

wml_ret = np.mean(wml['WML_ret'])*12*100
wml_std = np.std(wml['WML_ret'])*np.sqrt(12)*100
wml_sr = wml_ret/wml_std
wml_sk = wml['WML_log_ret'].skew()

# replication
table.loc[len(table.index)] = ('WML', wml_ret, wml_std, wml_sr, wml_sk)
table.set_index('decile').transpose()


## correlation with KD
# load Kent Daniel Data
dm = pd.read_csv('m_m_pt_tot.txt', sep='\s+', header=None)
dm.columns = 'date', 'decile', 'ret','a','b'
dm['date'] = pd.to_datetime(dm['date'], format='%Y%m%d', errors='ignore')
dm = dm[['date','decile','ret']]
DM_returns = pd.merge(dm, q3[['date','decile','DM_ret']], how='left', on=['date','decile']).dropna()

# compute decile portfolio correlation
dm_cor = DM_returns.groupby('decile')[['DM_ret','ret']].corr(method='pearson', min_periods=1).reset_index()
dm_cor = dm_cor[['decile','ret']].rename(columns={'ret':'DM_corr'})
dm_cor = dm_cor[dm_cor['DM_corr'] != 1].reset_index(drop=True)

# compute Daniel WML portfolio correlation
dm['WML_long'] = np.where(dm['decile']==10, 1, 0)
dm['WML_short'] = np.where(dm['decile']==1, -1, 0)
dm['dm_WML_ret'] = dm['WML_long']*dm['ret'] + dm['WML_short']*dm['ret']
dm_wml = dm.groupby('date')['dm_WML_ret'].sum().reset_index()
dm_wml = pd.merge(dm_wml, wml[['date','WML_ret']], how='left', on=['date'])
dm_wml_corr = dm_wml[['dm_WML_ret','WML_ret']].corr(method='pearson', min_periods=1)
dm_wml_corr = dm_wml_corr['WML_ret'][0]
dm_cor.loc[len(dm_cor.index)] = ('WML', dm_wml_corr)

# load FF data
ff_p.columns = 'date','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'
ff_p['date'] = pd.DataFrame(ff_p[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
ff_p['WML'] = ff_p['M10'] - ff_p['M1']
ff_p.drop(index=ff_p.index[-1],axis=0, inplace=True)
ff_p = ff_p.set_index('date')

# compute correlation with French's data
KRF_returns = q3[['date','decile','KRF_ret']]
KRF_returns = pd.pivot(KRF_returns, index='date', columns='decile')
KRF_returns = KRF_returns['KRF_ret']
KRF_returns.columns = 'M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'
KRF_returns['WML'] = KRF_returns['M10'] - KRF_returns['M1']
KRF_corr = KRF_returns.corrwith(ff_p, axis=0, method='pearson')

# final output
KRF_cor = [KRF_corr['M1'],KRF_corr['M2'],KRF_corr['M3'],KRF_corr['M4'],KRF_corr['M5'],KRF_corr['M6'],\
            KRF_corr['M7'],KRF_corr['M8'],KRF_corr['M9'],KRF_corr['M10'],KRF_corr['WML']]
dm_cor['KRF_corr'] = KRF_cor
corr = dm_cor.set_index('decile')
corr.transpose()
