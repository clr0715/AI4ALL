# Data Visualization資料視覺化
```
藉助於圖形化手段，
清晰有效地傳達與溝通訊息

https://zh.wikipedia.org/wiki/資料視覺化
```
# 資料視覺化の套件
```
Matplotlib(本課程使用)
Seaborn
Ggplot
Bokeh
Pyga
Plotly
```



# Google Colab上的範利
```
Charting in Colaboratory
https://colab.research.google.com/notebooks/charts.ipynb
```
### Line Plots
```
import matplotlib.pyplot as plt
 
x  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y1 = [1, 3, 5, 3, 1, 3, 5, 3, 1]
y2 = [2, 4, 6, 4, 2, 4, 6, 4, 2]
plt.plot(x, y1, label="line L")
plt.plot(x, y2, label="line H")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()
```

# MATPLOTLIB
```
官方網址 https://matplotlib.org/

使用指南  https://matplotlib.org/users/index.html

學習指南(Tutorials) https://matplotlib.org/tutorials/index.html
```
## MATPLOTLIB範例一
```
import numpy as np
import pylab as pl


# 產生資料
x = np.arange(0.0, 2.0*np.pi, 0.01)	
y = np.sin(x)			

#畫圖

pl.plot(x,y)		
pl.xlabel('x')			
pl.ylabel('y')
pl.title('sin')		
pl.show()
```
```
步驟一:先產生x軸的資料===使用陣列:0到2π之間，以0.01為step
x = np.arange(0.0, 2.0*np.pi, 0.01)    
 
步驟二:針對每一個x產生 y (y = sin(x))==== y 也是一個陣列
 
y = np.sin(x)

步驟三:畫圖==>設定圖形的呈現參數
pl.plot(x,y)	
....


步驟四:顯示圖形
pl.show()
```
# 延伸閱讀:推薦的教科書plot.ly

```
官方網址https://plot.ly/看看互動式資料視覺化成果
```
```
Python數據分析：基於Plotly的動態可視化繪圖
作者： 孫洋洋, 王碩, 邢夢來, 袁泉, 吳娜
電子工業出版社
https://github.com/sunshe35/PythonPlotlyCodes
```

### 延伸閱讀:書bokeh
```
官方網址  https://bokeh.pydata.org/en/latest/
!pip install bokeh

```
```
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

N = 4000

x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x).astype(int), np.floor(30+2*y).astype(int))]

output_notebook()
p = figure()
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
show(p)
```

## 延伸閱讀: seaborn
```
範例學習1:
https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb
```
```
範例學習2:
https://colab.research.google.com/drive/1o6MijFkNHiTPeS8Y5n59j2cH4-Mf2wX3
```
```
import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1});
```
```
https://www.data-insights.cn/?p=179
```

## 延伸閱讀:  altair
```
官方網址https://altair-viz.github.io/ 

```
```
import altair as alt
from vega_datasets import data
cars = data.cars()

alt.Chart(cars).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
).interactive()
```
