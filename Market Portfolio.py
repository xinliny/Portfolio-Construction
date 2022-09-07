import wrds
import pandas_datareader
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from decimal import Decimal


# LOAD DATA
load_data = False
data_folder = '/Users/Xinlin/Desktop/Quantitative Asset Management/Problem Sets/PS1/'
wrds_id = 'yxinlin'

min_shrcd = 10
max_shrcd = 11
min_year = 1925
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
    
    # save pkl
    crsp_raw.to_csv('crsp_raw.csv')
    dlret_raw.to_csv('dlret_raw.csv')

else:
    crsp_raw = pd.read_csv('crsp_raw.csv')
    dlret_raw = pd.read_csv('dlret_raw.csv')  


# LOAD FAMA-FRENCH 3 FACTOR MODEL DATA
load_data = False

if load_data:
    pd.set_option('precision', 4)
    ff3 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='1926', end='2022')
    ff3 = ff3.read()[0]/100
    ff3.columns = 'MktRF', 'SMB', 'HML', 'RF'
    ff3.to_csv('ff3.csv')

else:
    ff3 = pd.read_csv('ff3.csv')


# DATA CLEANING
crsp_raw['permno'] = crsp_raw['permno'].astype(int)
crsp_raw['date'] = pd.to_datetime(crsp_raw['date'], format='%Y-%m-%d', errors='ignore')
crsp_raw['date'] = pd.DataFrame(crsp_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
crsp_raw = crsp_raw.sort_values(by=['date', 'permno'])

dlret_raw.permno = dlret_raw.permno.astype(int)
dlret_raw['date'] = pd.to_datetime(dlret_raw['dlstdt'], format='%Y-%m-%d', errors='ignore')
dlret_raw['date'] = pd.DataFrame(dlret_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
dlret_raw = dlret_raw.sort_values(by=['date', 'permno'])

ff3['date'] = pd.to_datetime(ff3['Date'], format='%Y-%m-%d', errors='ignore')
ff3['date'] = pd.DataFrame(ff3[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
crsp = crsp_raw.merge(dlret_raw, how='outer', on=['date', 'permno'])


# COMPUTE ADJUSTED RETURNS
crsp['ret'] = np.where(crsp['ret'].notna() & crsp['dlret'].notna(), 
                       (1+crsp['ret'])*(1+crsp['dlret'])-1, crsp['ret'])
crsp['ret'] = np.where(crsp['ret'].isna() & crsp['dlret'].notna(),
                       crsp['dlret'], crsp['ret'])
crsp = crsp[crsp['ret'].notna()]
crsp = crsp[['date','permno','prc','shrout','ret']].sort_values(by=['date','permno']).reset_index(drop=True)

# COMPUTE MARKET EQUITY
crsp['me'] = crsp['prc'].abs() * crsp['shrout']

# FIND LAGGED MARKET EQUITY
crsp['new_date'] = crsp['date'] + dt.timedelta(days=20)
crsp['new_date'] = pd.DataFrame(crsp[['new_date']].values.astype('datetime64[ns]')) + MonthEnd(0)
lcrsp = crsp[['new_date','permno','me']].rename(columns={'new_date':'date', 'me':'lag_me'})
crsp = crsp[['date','permno','ret','me']]
crsp = crsp.merge(lcrsp, how='inner', on=['date','permno'])
crsp = crsp[crsp['ret'].notna() & crsp['lag_me'].notna()]
crsp = crsp.drop_duplicates(subset=['date','permno'])

# COMPUTE SUM OF LAGGED MARKET EQUITY AT EACH MONTH-END
sum_lag_me = crsp.groupby(['date'])['lag_me'].sum().reset_index().rename(columns={'lag_me':'stock_lag_mv'})
crsp = crsp.merge(sum_lag_me, how='outer', on='date')

# COMPUTE VALUE-WEIGHTED MARKET RETURNS
crsp['lag_weight'] = crsp['lag_me']/crsp['stock_lag_mv']
crsp['vw_ret'] = crsp['ret']*crsp['lag_weight']
vw_ret = crsp.groupby(['date'])['vw_ret'].sum().reset_index().rename(columns={'vw_ret': "stock_vw_ret"})

# COMPUTE EQUAL-WEIGHTED MARKET RETURNS
ew_ret = crsp.groupby(['date'])['ret'].mean().reset_index().rename(columns={'ret': "stock_ew_ret"})

monthly_crsp_stock = sum_lag_me.merge(ew_ret, how='inner', on=['date'])
monthly_crsp_stock = monthly_crsp_stock.merge(vw_ret, how='inner', on=['date'])
print(monthly_crsp_stock)


# PLOT CUMULATIVE VALUE-WEIGHTED & EQUAL-WEIGHTED RETURNS
vw_ret['cum_vw_ret'] = np.cumprod(1+vw_ret['stock_vw_ret']) - 1
ew_ret['cum_ew_ret'] = np.cumprod(1+ew_ret['stock_ew_ret']) - 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(vw_ret['cum_vw_ret'], label='c_vwret')
ax.plot(ew_ret['cum_ew_ret'], label='c_ewret')
ax1 = ax.twinx()
ax1.plot(sum_lag_me['stock_lag_mv'], label='stock_lag_mv', color='green')
fig.legend()
plt.show()


# COMPARISON WITH FAMA-FRENCH
df = pd.merge(vw_ret, ff3, how='outer', on=['date'])
df['ret_rep'] = df['stock_vw_ret'] - df['RF']
df = df[['date','ret_rep','MktRF']]
df = df[df['ret_rep'].notna() & df['MktRF'].notna()]

q2 = pd.DataFrame(columns=['Index', 'Replication', 'Fama-French'])

q2.loc[0, 'Index'] = 'Annualized Excess Return'
q2.loc[0, 'Replication'] = df['ret_rep'].mean()*12
q2.loc[0, 'Fama-French'] = df['MktRF'].mean()*12

q2.loc[1, 'Index'] = 'Annualized Standard Deviation'
q2.loc[1, 'Replication'] = df['ret_rep'].std()*np.sqrt(12)
q2.loc[1, 'Fama-French'] = df['MktRF'].std()*np.sqrt(12)

q2.loc[2, 'Index'] = 'Sharpe Ratio'
q2.loc[2, 'Replication'] = df['ret_rep'].mean()*12 / (df['ret_rep'].std()*np.sqrt(12))
q2.loc[2, 'Fama-French'] = df['MktRF'].mean()*12 / (df['MktRF'].std()*np.sqrt(12))

q2.loc[3, 'Index'] = 'Excess Skewness'
q2.loc[3, 'Replication'] = df['ret_rep'].skew()
q2.loc[3, 'Fama-French'] = df['MktRF'].skew()

q2.loc[4, 'Index'] = 'Excess Kurtosis'
q2.loc[4, 'Replication'] = df['ret_rep'].kurtosis()
q2.loc[4, 'Fama-French'] = df['MktRF'].kurtosis()

q2 = q2.set_index('Index')
print(q2)


# COMPUTE CORRELATION WITH FAMA-FRENCH DATA
cor = df[['ret_rep', 'MktRF']].corr(method='pearson', min_periods=1)
corr = Decimal(cor.iloc[0,1]).quantize(Decimal("0.00000001"), rounding = "ROUND_HALF_UP")
print(corr)
