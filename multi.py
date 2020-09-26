from ipdb import set_trace as br
import db, util, pandas as pd, numpy as np, scipy.stats, \
    matplotlib.pyplot as plt, seaborn as sns

nCol = 9
start = '20100101'
# tickers = db.get_HKMainBoard_tickers().to_list()
tickers = pd.read_csv('hsi_weights.csv').Ticker.to_list()
# for d in ['0566.HK','2840.HK']: tickers.remove(d)
nRow = len(tickers)//nCol
if len(tickers)%nCol>0: nRow += 1

closes = {t:db.get_close(t).loc[start:] for t in tickers[:nRow*nCol]}
lnchgs = {t:np.log(close).diff() for t,close in closes.items()}

rslt_raw = pd.DataFrame([], columns=['mu','sig','sharpe','N'])

fig, ax = plt.subplots(nRow,nCol, figsize=(25,12), sharex=True, sharey=True)
for r in range(nRow):
    for c in range(nCol):
        if r*nCol+c<len(tickers):
            ticker = tickers[r*nCol+c]
            ax[r,c].hist(lnchgs[ticker], bins=50, density=True)
            mu, sig = lnchgs[ticker].mean(), lnchgs[ticker].std()
            muA, sigA = mu*250, sig*np.sqrt(250)
            sharpeA = muA/sigA
            rslt_raw.loc[ticker] = (muA, sigA, sharpeA, len(lnchgs[ticker]))
            
            x=np.linspace(mu-3*sig, mu+3*sig, 100)
            ax[r,c].plot(x, scipy.stats.norm.pdf(x, mu, sig), c='r', lw=1)
            ax[r,c].axvline(0, c='k', lw=1)
            for i in range(-3,4):
                if i==0: lw = 1; color = 'g'
                else: lw = abs(1/i); color = 'r'
                ax[r,c].axvline(mu+i*sig, c=color, ls='--', lw=lw)
            title = (f'{ticker}:{db.get_Full_NameList().loc[ticker,"Name_Alt"]}, N:{len(lnchgs[ticker])}\n'
                        f'平均:{muA:+.2%} 差:{sigA:.2%} #:{sharpeA:+.2%}')
            ax[r,c].set_title(title, fontproperties=util.get_chinese_font(size=8))
            ax[r,c].set_xlim(-.08,+.08)
            ax[r,c].set_ylim(0,50)

def display(df, N=20):
    df_display = df.applymap(lambda s:f'{s:.1%}').head(N)
    print(df_display)
    return df_display
rslt_by_mu = rslt_raw.sort_values(by='mu', ascending=False)
rslt_by_sig = rslt_raw.sort_values(by='sig', ascending=False)
rslt_by_sharpe = rslt_raw.sort_values(by='sharpe', ascending=False)

br()
plt.tight_layout()
plt.show()
br()