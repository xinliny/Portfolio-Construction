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

load_data = False
data_folder = '/Users/Xinlin/Desktop/Quantitative Asset Management/Problem Sets/PS2/'
wrds_id = 'yxinlin'

min_year = 1926
max_year = 2021

if load_data:
    conn = wrds.Connection(wrds_username=wrds_id)
    # load CRSP bond returns
    bond_raw = conn.raw_sql("""
                          select kycrspid, mcaldt, tmretnua, tmtotout
                          from crsp.tfz_mth
                          where mcaldt between '01/01/"""+str(min_year)+"""' and '12/31/"""+str(max_year)+"""'
                          """)
    
    # load CRSP risk-free rates
    rf_raw = conn.raw_sql("""
                          select caldt, t90ret, t30ret
                          from crsp.mcti
                          where caldt between '01/01/"""+str(min_year)+"""' and '12/31/"""+str(max_year)+"""'
                          """)
    conn.close()
    
    # save csv
    bond_raw.to_csv('bond_raw.csv')
    rf_raw.to_csv('rf_raw.csv')

else:
    bond_raw = pd.read_csv('bond_raw.csv')
    rf_raw = pd.read_csv('rf_raw.csv')                      

    
# DATA CLEANING
bond_raw['date'] = pd.to_datetime(bond_raw['mcaldt'], format='%Y-%m-%d', errors='ignore')
bond_raw['date'] = pd.DataFrame(bond_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
bond_raw = bond_raw.sort_values(by=['date', 'kycrspid'])
bond = bond_raw[bond_raw['tmretnua'].notna() & bond_raw['tmtotout'].notna()]

# COMPUTE EQUAL-WEIGHTED RETURN
bond_ew_ret = bond.groupby('date')['tmretnua'].mean().reset_index().rename(columns={'tmretnua':'bond_ew_ret'})

# COMPUTE VALUE-WEIGHTED RETURN
bond['new_date'] = bond['date'] + dt.timedelta(days=20)
bond['new_date'] = pd.DataFrame(bond[['new_date']]) + MonthEnd(0)
lbond = bond[['new_date','kycrspid','tmtotout']].rename(columns={'new_date':'date', 'tmtotout':'lag_amount'})
bond = bond[['date','kycrspid','tmretnua','tmtotout']]
bond = bond.merge(lbond, how='inner', on=['date','kycrspid'])
bond = bond[bond['tmretnua'].notna() | bond['lag_amount'].notna()]
bond = bond.drop_duplicates(subset=['date','kycrspid'])

bond['sum_lag_amount'] = bond.groupby('date')['lag_amount'].transform('sum')
bond['weight'] = bond['lag_amount']/bond['sum_lag_amount']

bond['vw_ret'] = bond['weight'] * bond['tmretnua']
bond_vw_ret = bond.groupby('date')['vw_ret'].sum().reset_index().rename(columns={'vw_ret':'bond_vw_ret'})

# COMPUTE BOND MARKET VALUE
price = 1000
bond['lag_mv'] = bond['lag_amount']*price
bond_lag_mv = bond.groupby('date')['lag_mv'].sum().reset_index().rename(columns={'lag_mv':'bond_lag_mv'})


# DATA CLEANING FOR RISK-FREE RATES DATA
rf_raw['date'] = pd.to_datetime(rf_raw['caldt'], format='%Y-%m-%d', errors='ignore')
rf_raw['date'] = pd.DataFrame(rf_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
rf_raw = rf_raw[rf_raw['t90ret'].notna() & rf_raw['t30ret'].notna()]
monthly_crsp_riskless = rf_raw[['date','t30ret','t90ret']]

# USE 'MARKET PORTFOLIO' OUTPUT
monthly_crsp_stock = pd.read_csv('monthly_crsp_stock.csv', index_col=False)
monthly_crsp_stock = monthly_crsp_stock[['date','stock_lag_mv','stock_ew_ret','stock_vw_ret']]
monthly_crsp_stock['date'] = pd.to_datetime(monthly_crsp_stock['date'], format='%Y-%m-%d', errors='ignore')

df = monthly_crsp_stock.merge(monthly_crsp_bond, how='inner', on=['date'])
df = df.merge(monthly_crsp_riskless, how='inner', on=['date'])

# COMPUTE EXCESS RETURN FOR STOCK AND BOND
df['stock_excess_vw_ret'] = df['stock_vw_ret'] - df['t30ret']
df['bond_excess_vw_ret'] = df['bond_vw_ret'] - df['t30ret']
monthly_crsp_universe = df[['date','stock_lag_mv','stock_excess_vw_ret','bond_lag_mv','bond_excess_vw_ret','t30ret']]


# COMPUTE INVERSE VOLATILITY OF PAST 36-MONTH & CUMULATIVE MONTHLY EXCESS RETURN
S = []
B = []
P = []
for i in range(0+36, 1151):
    temp = port[['date','stock_excess_vw_ret','bond_excess_vw_ret']].iloc[i-36:i]
    s = np.std(temp['stock_excess_vw_ret'])
    b = np.std(temp['bond_excess_vw_ret'])
    S = np.append(S, s)
    B = np.append(B, b)
    
for i in range(0+36, 1151):
    temp1 = port[['date','excess_vw_ret']].iloc[0:i]
    p = np.std(temp1['excess_vw_ret'])
    P = np.append(P, p)


# RISK-PARITY UNLEVERED PORTFOLIO WEIGHT
sigma = pd.DataFrame(data={'date':port['date'].iloc[36:1151], 'stock_sigma':S, 'stock_inv_sigma':1/S, 
                           'bond_sigma':B ,'bond_inv_sigma':1/B, 'vwport_sigma':P, 'vwport_inv_sigma':1/P})
sigma['unlevered_k'] = 1 / (sigma['stock_inv_sigma'] + sigma['bond_inv_sigma'])
sigma['s_w'] = sigma['unlevered_k']*sigma['stock_inv_sigma']
sigma['b_w'] = sigma['unlevered_k']*sigma['bond_inv_sigma']

sigma['test_k'] = 1
sigma['test_s_lw'] = sigma['test_k']*sigma['stock_inv_sigma']
sigma['test_b_lw'] = sigma['test_k']*sigma['bond_inv_sigma']


# RISK-PARITY UNLEVERED PORTFOLIO EXCESS RETURN
rp = monthly_crsp_universe.copy()
rp = rp.merge(sigma, how='inner', on=['date'])
rp['excess_unlevered_rp_ret'] = rp['s_w']*rp['stock_excess_vw_ret'] + rp['b_w']*rp['bond_excess_vw_ret']


# RISK-PARITY LEVERED PORTFOLIO EXCESS RETURN
rp['test_rf_w'] = rp['test_s_lw'] + rp['test_b_lw'] - 1
rp['test_levered_rp_ret'] = rp['test_s_lw']*rp['stock_excess_vw_ret'] + rp['test_b_lw']*rp['bond_excess_vw_ret']- rp['test_rf_w']*rp['t30ret']
rp['levered_k'] = np.std(port['excess_vw_ret']) / np.std(rp['test_levered_rp_ret'])

rp['s_lw'] = rp['levered_k']*rp['stock_inv_sigma']
rp['b_lw'] = rp['levered_k']*rp['bond_inv_sigma']
rp['rf_w'] = rp['s_lw'] + rp['b_lw'] - 1
rp['excess_levered_rp_ret'] = rp['s_lw']*rp['stock_excess_vw_ret'] + rp['b_lw']*rp['bond_excess_vw_ret']- rp['rf_w']*rp['t30ret']
