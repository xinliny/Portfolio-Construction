import wrds
import datetime as dt
import pandas as pd
import numpy as np
import pandas_datareader
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd


# load data from WRDS

data_folder = '/Users/Xinlin/Desktop/Quantitative Asset Management/Problem Sets/PS4/'
wrds_id = 'yxinlin'

## Compustat (data from wrds)
conn = wrds.Connection(wrds_username=id_wrds)
cstat = conn.raw_sql("""
                    select a.gvkey, a.datadate, a.at, a.pstkl, a.txditc, a.fyear, a.ceq, a.lt, 
                    a.mib, a.itcb, a.txdb, a.pstkrv, a.seq, a.pstk, b.sic, b.year1, b.naics
                    from comp.funda as a
                    left join comp.names as b
                    on a.gvkey = b.gvkey
                    where indfmt='INDL'
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    """)

## Pension data               
Pension = conn.raw_sql("""
                        select gvkey, datadate, prba
                        from comp.aco_pnfnda
                        where indfmt='INDL'
                        and datafmt='STD'
                        and popsrc='D'
                        and consol='C'
                        """)

## CRSP-Compustat link table (data from wrds)
crsp_link = conn.raw_sql("""
                          select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, liid,
                          linkdt, linkenddt
                          from crspq.ccmxpf_linktable
                          where substr(linktype,1,1)='L'
                          and (linkprim ='C' or linkprim='P')
                          """)

## CRSP returns (data from wrds)
ps4_ret = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.exchcd, b.siccd, b.naics,
                      a.ret, a.retx, a.shrout, a.prc
                      from crspq.msf as a
                      left join crspq.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where b.shrcd in (10,11)
                      and b.exchcd in (1,2,3)
                      """)

ps4_dlret = conn.raw_sql("""
                        select a.permno, a.permco, a.dlret, a.dlretx, a.dlstdt, 
                        b.exchcd as dlexchcd, b.siccd as dlsiccd, b.naics as dlnaics
                        from crspq.msedelist as a
                        left join crspq.msenames as b
                        on a.permno=b.permno
                        and b.namedt<=a.dlstdt
                        and a.dlstdt<=b.nameendt
                        where b.shrcd in (10,11)
                        and b.exchcd in (1,2,3)
                        """) 

conn.close()

## Fama and French 3 Factors
pd.set_option('precision', 2)
data2 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',
                                                      start='1900', end=str(dt.datetime.now().year+1))
french = data2.read()[0] / 100 # Monthly data
french['Mkt'] = french['Mkt-RF'] + french['RF']

## Book-to-Market Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_BE-ME',
                                                      start='1900', end=str(dt.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 
                         'Dec 4', 'Dec 5', 'Dec 6', 
                         'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'BM01','BM02','BM03','BM04','BM05','BM06','BM07','BM08','BM09','BM10'
french = pd.merge(french,data2,how='left',on=['Date'])

## Size Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_ME',
                                                      start='1900', end=str(dt.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 
                         'Dec 4', 'Dec 5', 'Dec 6', 
                         'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'ME01','ME02','ME03','ME04','ME05','ME06','ME07','ME08','ME09','ME10'
french = pd.merge(french,data2,how='left',on=['Date'])

## 25 Book-to-Market and Size Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('25_Portfolios_5x5',
                                                      start='1900', end=str(dt.datetime.now().year+1))
data2 = data2.read()[0].rename(columns={"SMALL LoBM":"ME1 BM1",
                                        "SMALL HiBM":"ME1 BM5",
                                        "BIG LoBM":"ME5 BM1",
                                        "BIG HiBM":"ME5 BM5"}) / 100
french = pd.merge(french,data2,how='left',on=['Date'])

## Changing date format and save
french = french.reset_index().rename(columns={"Date":"date"})
french['date'] = pd.DataFrame(french[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)

## save csv files
cstat.to_csv(data_folder + 'cstat.csv')
Pension.to_csv(data_folder + 'Pension.csv')
crsp_link.to_csv(data_folder + 'crsp_link.csv')
ps4_ret.to_csv(data_folder + 'ps4_ret.csv')
ps4_dlret.to_csv(data_folder + 'ps4_dlret.csv')
french.to_csv(data_folder + 'french.csv')

## read csv
cstat = pd.read_csv('cstat.csv')
Pension = pd.read_csv('Pension.csv')
crsp_link = pd.read_csv('crsp_link.csv')
crsp_raw = pd.read_csv('ps4_ret.csv')
dlret_raw = pd.read_csv('ps4_dlret.csv')
FF = pd.read_csv('french.csv')

# Calculate book equity

## clean data
cstat['datadate'] = pd.to_datetime(cstat['datadate'])
Pension['datadate'] = pd.to_datetime(Pension['datadate'])
compustat = cstat.merge(Pension, how = 'outer', on = ['gvkey','datadate'])

## shareholders' equtiy
compustat['SHE'] = np.where(compustat['seq'].notna(), compustat['seq'], 
                            compustat['ceq'] + compustat['pstk'])
compustat['SHE'] = np.where(compustat['SHE'].notna(), compustat['SHE'], 
                            compustat['at'] - compustat['lt'] - compustat['mib'])
compustat['SHE'] = np.where(compustat['SHE'].notna(), compustat['SHE'], 
                            compustat['at'] - compustat['lt'])

## deferred taxes investment tax credit
compustat['itcb'] = compustat['itcb'].fillna(0)
compustat['txdb'] = compustat['txdb'].fillna(0)
compustat['DT'] = np.where(compustat['txditc'].notna(), compustat['txditc'],
                           compustat['itcb'] + compustat['txdb'])

## Book value of preferred stock
compustat['PS'] = np.where(compustat['pstkrv'].notna(), compustat['pstkrv'], compustat['pstkl'])
compustat['PS'] = np.where(compustat['PS'].notna(), compustat['PS'], compustat['pstk'])
compustat['PS'] = compustat['PS'].fillna(0)

## book equity
compustat['prba'] = compustat['prba'].fillna(0)
compustat['BE'] = compustat['SHE'] - compustat['PS'] + compustat['DT'] - compustat['prba']
compustat['BE'] = np.where(compustat['BE']>0, compustat['BE'], np.nan)

# Calculate market equity

## clean data
crsp_raw[['permno','permco']] = crsp_raw[['permno','permco']].astype(int)
crsp_raw['date'] = pd.to_datetime(crsp_raw['date'], format='%Y-%m-%d', errors='ignore')
crsp_raw['date'] = pd.DataFrame(crsp_raw['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
crsp_raw = crsp_raw.sort_values(by=['permno', 'date'])

dlret_raw[['permno','permco']] = dlret_raw[['permno','permco']].astype(int)
dlret_raw['date'] = pd.to_datetime(dlret_raw['dlstdt'], format='%Y-%m-%d', errors='ignore')
dlret_raw['date'] = pd.DataFrame(dlret_raw['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
dlret_raw = dlret_raw.sort_values(by=['permno', 'date'])

## merge two returns 
crsp = crsp_raw.merge(dlret_raw[['permno','permco','date','dlret']], 
                      how='outer', on=['date','permco','permno'])
crsp['ret'] = np.where(crsp['ret'].notna() & crsp['dlret'].notna(), 
                        (1+crsp['ret'])*(1+crsp['dlret'])-1, crsp['ret'])
crsp['ret'] = np.where(crsp['ret'].isna() & crsp['dlret'].notna(), crsp['dlret'], crsp['ret'])
crsp = crsp.sort_values(by=['permno', 'date'])

## market equity
crsp['me'] = crsp['prc'].abs() * crsp['shrout']
crsp_summe = crsp.groupby(['date','permco'])['me'].sum().reset_index()

dec_me = crsp_summe[crsp_summe['date'].dt.month == 12]
dec_me['Year'] = dec_me['date'].dt.year
dec_me['FYear'] = dec_me['Year'] + 1
dec_me = dec_me.rename(columns = {'me':'lag_dec_me'})

jun_me = crsp_summe[crsp_summe['date'].dt.month == 6]
jun_me['FYear'] = jun_me['date'].dt.year
jun_me = jun_me.rename(columns = {'me':'jun_me'})

crsp['FYear'] = crsp['date'].dt.year
crsp_me = crsp[crsp['date'].dt.month == 6].merge(dec_me[['FYear','permco','lag_dec_me']],
                                                 how = 'outer', on = ['FYear','permco'])
crsp_me = crsp_me.merge(jun_me[['FYear','permco','jun_me']], 
                        how = 'outer', on = ['FYear','permco'])
crsp_me = crsp_me[['date','FYear','permno','permco','exchcd','ret','retx','jun_me','lag_dec_me']]
crsp_me = crsp_me[crsp_me['date'].notna()].reset_index(drop=True)

# link crsp & compustat
crsp_link['linkenddt'] = crsp_link['linkenddt'].fillna(pd.to_datetime('today'))
crsp_link['linkenddt'] = pd.to_datetime(crsp_link['linkenddt'], format='%Y-%m-%d', errors='ignore')
crsp_link['linkdt'] = pd.to_datetime(crsp_link['linkdt'], format='%Y-%m-%d', errors='ignore')
link_data = compustat[['gvkey','datadate','BE']].merge(crsp_link, how = 'outer', on = ['gvkey'])
link_data['date'] = link_data['datadate'] + MonthEnd(6)
link_data = link_data[(link_data['date'] <= link_data['linkenddt']) & 
                      (link_data['date'] >= link_data['linkdt'])].reset_index(drop=True)
link_data = link_data[['datadate','date','gvkey','permno','permco','BE']]
ccm = crsp_me.merge(link_data[['date','permno','BE']], how = 'inner', on = ['date','permno'])

ccm['BE_ME'] = ccm['BE']*1000 / ccm['lag_dec_me']
ccm = ccm[(ccm['BE_ME'].notna()) & (ccm['jun_me'].notna())].reset_index(drop=True)
ccm = ccm.sort_values(by=['permno', 'date'])

# portfolio decile

ccm['BEME_decile'] = np.nan
ccm['ME_decile'] = np.nan

def nyse_brkpts(exc, ranking, K):
    loc_nyse = (exc == 1) # the locations of nyse stocks 
    breakpoints = pd.qcut(ranking[loc_nyse], K, retbins=True, labels=False)[1]
    breakpoints[0] = -np.inf
    breakpoints[K] = np.inf
    return pd.cut(ranking, bins=breakpoints, labels=False) + 1

all_year = ccm['FYear'].unique()
for y in all_year:
    sub_ccm = ccm[ccm['FYear'] == y]
    decile_beme = nyse_brkpts(sub_ccm['exchcd'], sub_ccm['BE_ME'], 10)
    decile_me = nyse_brkpts(sub_ccm['exchcd'], sub_ccm['jun_me'], 10)
    ccm.loc[sub_ccm.index, 'BEME_decile'] = decile_beme
    ccm.loc[sub_ccm.index, 'ME_decile'] = decile_me

# 10 size & 10 B/E portfolio vw returns

crsp_ret = crsp[['date','permno','permco','ret','me']]
crsp_ret['fdate'] = crsp_ret['date'] - MonthEnd(6)
crsp_ret['FYear'] = crsp_ret['fdate'].dt.year
crsp_ret = crsp_ret.merge(ccm[['FYear','permno','BEME_decile','ME_decile']],
                          how = 'inner', on = ['FYear','permno'])
crsp_ret['lag_me'] = crsp_ret.groupby(['permno'])['me'].shift(1)
crsp_ret = crsp_ret[crsp_ret['lag_me'].notna()].reset_index(drop=True)

crsp_ret['lag_summe_me'] = crsp_ret.groupby(['ME_decile','date'])['lag_me'].transform('sum')
crsp_ret['weighted_ret_me'] = (crsp_ret['lag_me'] / crsp_ret['lag_summe_me']) * crsp_ret['ret']
Size_Ret = crsp_ret.groupby(['date','ME_decile'])['weighted_ret_me'].sum().reset_index()

crsp_ret['lag_summe_beme'] = crsp_ret.groupby(['BEME_decile','date'])['lag_me'].transform('sum')
crsp_ret['weighted_ret_beme'] = (crsp_ret['lag_me'] / crsp_ret['lag_summe_beme']) * crsp_ret['ret']
BtM_Ret = crsp_ret.groupby(['date','BEME_decile'])['weighted_ret_beme'].sum().reset_index()

Size_Ret = Size_Ret[(Size_Ret['date'].dt.year < 2022) &
                    (Size_Ret['date'].dt.year > 1972)].reset_index(drop=True)
BtM_Ret = BtM_Ret[(BtM_Ret['date'].dt.year < 2022) &
                  (BtM_Ret['date'].dt.year > 1972)].reset_index(drop=True)

# SMB & HML

# 6 portfolios
ccm['FF3_size'] = np.nan
ccm['FF3_BEME'] = np.nan

for y in all_year:
    
    sub_ccm = ccm[ccm['FYear'] == y]
    
    FF3_size = nyse_brkpts(sub_ccm['exchcd'], sub_ccm['jun_me'], 2)
    ccm.loc[sub_ccm.index, 'FF3_size'] = FF3_size
    
    loc_nyse = (sub_ccm['exchcd'] == 1)
    decile_beme30 = np.quantile(sub_ccm['BE_ME'][loc_nyse],0.3)
    decile_beme70 = np.quantile(sub_ccm['BE_ME'][loc_nyse],0.7)
    beme_bins = [-np.inf, decile_beme30, decile_beme70, np.inf]
    ccm.loc[sub_ccm.index, 'FF3_BEME'] = pd.cut(sub_ccm['BE_ME'], 
                                                bins=beme_bins, labels=False) + 1

def FF3_fac(size,be_me):
    if size == 1:
        if be_me == 1:
            fac = 1 # small growth
        if be_me == 2:
            fac = 2 # small neutral
        if be_me == 3:
            fac = 3 # small value
    else:
        if be_me == 1:
            fac = 4 # big growth
        if be_me == 2:
            fac = 5 # big neutral
        if be_me == 3:
            fac = 6 # big value
    return fac
    
ccm['6port'] = ccm.apply(lambda x: FF3_fac(x['FF3_size'],x['FF3_BEME']),axis=1)

# vwret of 6 portfolios
FF3_ret = crsp[['date','permno','permco','ret','me']]
FF3_ret['fdate'] = FF3_ret['date'] - MonthEnd(6)
FF3_ret['FYear'] = FF3_ret['fdate'].dt.year
FF3_ret = FF3_ret.merge(ccm[['FYear','permno','6port']],
                        how = 'inner', on = ['FYear','permno'])
FF3_ret['lag_me'] = FF3_ret.groupby(['permno'])['me'].shift(1)
FF3_ret = FF3_ret[FF3_ret['lag_me'].notna()].reset_index(drop=True)
    
FF3_ret['lag_summe'] = FF3_ret.groupby(['6port','date'])['lag_me'].transform('sum')
FF3_ret['weighted_ret'] = (FF3_ret['lag_me'] / FF3_ret['lag_summe']) * FF3_ret['ret']
FF_6port_vwret = FF3_ret.groupby(['date','6port'])['weighted_ret'].sum().reset_index()
FF_6port_vwret = FF_6port_vwret[(FF_6port_vwret['date'].dt.year < 2022) &
                                (FF_6port_vwret['date'].dt.year > 1972)].reset_index(drop=True)

smal_grow = FF_6port_vwret[FF_6port_vwret['6port'] == 1]['weighted_ret'].reset_index(drop=True)
smal_neu = FF_6port_vwret[FF_6port_vwret['6port'] == 2]['weighted_ret'].reset_index(drop=True)
smal_val = FF_6port_vwret[FF_6port_vwret['6port'] == 3]['weighted_ret'].reset_index(drop=True)
big_grow = FF_6port_vwret[FF_6port_vwret['6port'] == 4]['weighted_ret'].reset_index(drop=True)
big_neu = FF_6port_vwret[FF_6port_vwret['6port'] == 5]['weighted_ret'].reset_index(drop=True)
big_val = FF_6port_vwret[FF_6port_vwret['6port']== 6]['weighted_ret'].reset_index(drop=True)
date = FF_6port_vwret[FF_6port_vwret['6port'] == 1]['date'].reset_index(drop=True)

# SMB & HML exret
SMB = (1/3) * (smal_grow + smal_neu + smal_val - big_grow - big_neu - big_val)
HML = (1/2) * (smal_val + big_val - smal_grow - big_grow)
my_FF3 = pd.DataFrame({'date':date, 'SMB':SMB, 'HML':HML})
risk_free = FF[['date','RF']]
risk_free['date'] = pd.to_datetime(risk_free['date'])
my_FF3 = my_FF3.merge(risk_free, how='inner', on=['date'])
my_FF3['SMB'] = my_FF3['SMB'] - my_FF3['RF']
my_FF3['HML'] = my_FF3['HML'] - my_FF3['RF']

# merge all returns
Q1 = pd.concat([Size_Ret, BtM_Ret.drop(['date'], axis=1)], axis=1)
Q1 = Q1.merge(my_FF3, how = 'outer', on = ['date'])
Q1['Size_Ret'] = Q1['weighted_ret_me'] - Q1['RF']
Q1['BtM_Ret'] = Q1['weighted_ret_beme'] - Q1['RF']
Q1['Year'] = Q1['date'].dt.year
Q1['Month'] = Q1['date'].dt.month
Q1 = Q1[['date','Year','Month','ME_decile','Size_Ret',
         'BEME_decile','BtM_Ret','HML','SMB']]

# Statistics of 10 size & 10 B/E portfolio returns 

FF['date'] = pd.to_datetime(FF['date'])
FF = FF[(FF['date'].dt.year < 2022) &
        (FF['date'].dt.year > 1972)].reset_index(drop=True)

corr = pd.DataFrame(columns=['Size','BM'], index=np.arange(1,11,1))
size_charac = pd.DataFrame(columns=['Annual_EXRet(%)','Annual_vol(%)','SR','Skew'],
                           index=np.arange(1,11,1))
bm_charac = pd.DataFrame(columns=['Annual_EXRet(%)','Annual_vol(%)','SR','Skew'],
                         index=np.arange(1,11,1))

for i in range(0,10):
    
    i = int(i)
    
    my_size_ret = Q1[Q1['ME_decile'] == (i+1)]['Size_Ret'].reset_index(drop=True)
    size_charac.loc[i+1,'Annual_EXRet(%)'] = my_size_ret.mean() * 1200
    size_charac.loc[i+1,'Annual_vol(%)'] = my_size_ret.std() * np.sqrt(12) * 100
    size_charac.loc[i+1,'SR'] = my_size_ret.mean() * 12 / (my_size_ret.std() * np.sqrt(12))
    size_charac.loc[i+1,'Skew'] = my_size_ret.skew()
    FF_size_ret = FF.iloc[:, (i+17)]
    corr.loc[i+1, 'Size'] = np.corrcoef(my_size_ret, FF_size_ret)[0,1]
    
    my_bm_ret = Q1[Q1['BEME_decile'] == (i+1)]['BtM_Ret'].reset_index(drop=True)
    bm_charac.loc[i+1,'Annual_EXRet(%)'] = my_bm_ret.mean() * 1200
    bm_charac.loc[i+1,'Annual_vol(%)'] = my_bm_ret.std() * np.sqrt(12) * 100
    bm_charac.loc[i+1,'SR'] = my_bm_ret.mean() * 12 / (my_bm_ret.std() * np.sqrt(12))
    bm_charac.loc[i+1,'Skew'] = my_bm_ret.skew()
    FF_bm_ret = FF.iloc[:, (i+7)]
    corr.loc[i+1, 'BM'] = np.corrcoef(my_bm_ret, FF_bm_ret)[0,1]
    
# long-short portfolios
size_ls = Q1[Q1['ME_decile'] == 1]['Size_Ret'].reset_index(drop=True) -\
          Q1[Q1['ME_decile'] == 10]['Size_Ret'].reset_index(drop=True)
bm_ls = Q1[Q1['BEME_decile'] == (10)]['BtM_Ret'].reset_index(drop=True) -\
        Q1[Q1['BEME_decile'] == (1)]['BtM_Ret'].reset_index(drop=True)
        
size_ls_mean = size_ls.mean() * 1200
size_ls_std = size_ls.std() * np.sqrt(12) * 100
size_ls_sr = size_ls_mean / size_ls_std
size_ls_skew = size_ls.skew()
bm_ls_mean = bm_ls.mean() * 1200
bm_ls_std = bm_ls.std() * np.sqrt(12) * 100
bm_ls_sr = bm_ls_mean / bm_ls_std
bm_ls_skew = bm_ls.skew()

size_ls_df = pd.DataFrame({'Annual_EXRet(%)':size_ls_mean,
                           'Annual_vol(%)':size_ls_std,
                           'SR':size_ls_sr,
                           'Skew':size_ls_skew},
                          index = ['long_short'])
bm_ls_df = pd.DataFrame({'Annual_EXRet(%)':bm_ls_mean,
                         'Annual_vol(%)':bm_ls_std,
                         'SR':bm_ls_sr,
                         'Skew':bm_ls_skew},
                        index = ['long_short'])

size_charac = pd.concat([size_charac, size_ls_df])
bm_charac = pd.concat([bm_charac, bm_ls_df])

# plot the figure

# ME
plt.figure()
for i in range(1,11):
    decile = Q1[Q1['ME_decile'] == i]
    decile = decile[decile.Year > 2010]
    decile_logret = np.log(decile['Size_Ret'] + 1)
    decile_cumret = decile_logret.cumsum()
    plt.plot(decile['date'], decile_cumret, label=str(i))
plt.legend()
plt.title('Size decile portfolio cumulative log returns')
plt.show()

# BE/ME
plt.figure()
for i in range(1,11):
    decile = Q1[Q1['BEME_decile'] == i]
    decile = decile[decile.Year > 2010]
    decile_logret = np.log(decile['BtM_Ret'] + 1)
    decile_cumret = decile_logret.cumsum()
    plt.plot(decile['date'], decile_cumret, label=str(i))
plt.legend()
plt.title('Book-to-Market decile portfolio cumulative log returns')
plt.show()

# Correlations between my replication and the data from the website

corr_ff3 = pd.DataFrame(index=['HML','SMB'], columns=['correlation'])
ff3_charac = pd.DataFrame(columns=['Annual_EXRet(%)','Annual_vol(%)','SR','Skew'],
                          index=['HML','SMB'])

corr_ff3.loc['HML','correlation'] = np.corrcoef(FF['HML'], my_FF3['HML'])[0,1]
corr_ff3.loc['SMB','correlation'] = np.corrcoef(FF['SMB'], my_FF3['SMB'])[0,1]

ff3_charac.loc['HML','Annual_EXRet(%)'] = my_FF3['HML'].mean() * 1200
ff3_charac.loc['HML','Annual_vol(%)'] = my_FF3['HML'].std() * np.sqrt(12) * 100
ff3_charac.loc['HML','SR'] = my_FF3['HML'].mean() * 12 / (my_FF3['HML'].std() * np.sqrt(12))
ff3_charac.loc['HML','Skew'] = my_FF3['HML'].skew()

ff3_charac.loc['SMB','Annual_EXRet(%)'] = my_FF3['SMB'].mean() * 1200
ff3_charac.loc['SMB','Annual_vol(%)'] = my_FF3['SMB'].std() * np.sqrt(12) * 100
ff3_charac.loc['SMB','SR'] = my_FF3['SMB'].mean() * 12 / (my_FF3['SMB'].std() * np.sqrt(12))
ff3_charac.loc['SMB','Skew'] = my_FF3['SMB'].skew()

# plot the figure
plt.figure()
plt.plot(my_FF3['date'], my_FF3['SMB'], label='SMB')
plt.plot(my_FF3['date'], my_FF3['HML'], label='HML', color='orange')
plt.legend()
plt.show()
