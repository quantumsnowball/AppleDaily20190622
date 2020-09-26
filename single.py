from ipdb import set_trace as br
import db, util, sys, pandas as pd, numpy as np, scipy.stats, \
    matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter

ticker = sys.argv[1]
start = '20100101'
close = db.get_close(ticker).loc[start:]
lnchg = np.log(close).diff()

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.hist(lnchg, bins=100, density=True)
mu, sig = lnchg.mean(), lnchg.std()
muA, sigA = mu*250, sig*np.sqrt(250)
x=np.linspace(mu-3*sig, mu+3*sig, 100)
ax.plot(x, scipy.stats.norm.pdf(x, mu, sig), c='r', lw=1)
ax.axvline(0, c='k', lw=3)
for i in range(-3,4):
    if i==0: lw = 3; c = 'g'
    else: lw = abs(3/i); c = 'r'
    ax.axvline(mu+i*sig, c=c, ls='--', lw=lw)
title = (f'{ticker}:{db.get_Full_NameList().loc[ticker,"Name_Alt"]}, 取樣(N):{len(lnchg)}日\n'
    f'平均(年化):{muA:+,.2%} 標準差(年化):{sigA:.2%} 夏普比率(Sharpe Ratio):{muA/sigA:+,.2%}')
ax.set_title(title, fontproperties=util.get_chinese_font(size=12))
ax.set_xlabel('日回報率(%)', fontproperties=util.get_chinese_font(size=9))
ax.set_ylabel('機率密度(Probability Density)', fontproperties=util.get_chinese_font(size=9))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1%}'))
ax.set_xlim(-.08,+.08)
# ax.set_xlim(-.005,+.005)
ax.set_ylim(0,50)

plt.show()