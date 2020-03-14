# 1.Data Visualization資料視覺化
```
藉助於圖形化手段，
清晰有效地傳達與溝通訊息

https://zh.wikipedia.org/wiki/資料視覺化
```
# 2.資料視覺化の套件
```
Matp2lotlib(本課程使用)
Seaborn
Ggplot
Bokeh
Pyga
Plotly
```
# 3.Google Colab上的範利
```
Charting in Colaboratory
https://colab.research.google.com/notebooks/charts.ipynb
```
### Line Plots折線圖:基本統計圖形
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

# 4.MATPLOTLIB
```
官方網址 https://matplotlib.org/

使用指南  https://matplotlib.org/users/index.html

學習指南(Tutorials) https://matplotlib.org/tutorials/index.html
```
# 4.1.MATPLOTLIB範例一
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
### matplotlib.pyplot
```
https://matplotlib.org/api/pyplot_summary.html
```
### plot()函數 matplotlib.pyplot.plot
```
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

使用plot()函數畫圖

pl.plot(t,s)            #畫圖，以t為橫坐標，s為縱坐標
pl.xlabel('x')            #設定坐標軸標籤
pl.ylabel('y')
pl.title('sin')        #設定圖形標題
pl.show()                #顯示圖形
```
### plot()函數範例:看看底下產生的數學公式
```
https://matplotlib.org/gallery/pyplots/pyplot_mathtext.html#sphx-glr-gallery-pyplots-pyplot-mathtext-py
```
```
import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)

plt.plot(t, s)
plt.title(r'$\alpha_i > \beta_i$', fontsize=20)

plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
         fontsize=20)
         
plt.xlabel('time (s)')
plt.ylabel('volts (mV)')
plt.show()
```
# 4.2.基本統計圖形: 散佈圖 (Scatter plot)  
```
用途:看看資料有何關係??
```
```
https://en.wikipedia.org/wiki/Scatter_plot

散佈圖是一種使用笛卡兒坐標來顯示一組數據的通常兩個變量的值的圖或數學圖。
如果對點進行了編碼，則可以顯示一個附加變量。
數據顯示為點的集合，每個點具有確定水平軸上位置的一個變量的值和確定垂直軸上位置的另一個變量的值。
```

# 範例練習二: 散佈圖 (Scatter plot)範例一
```
import numpy as np
import pylab as pl

# 產生資料
x = np.arange(0, 2.0*np.pi, 0.2)
y = np.cos(x)

#畫圖
pl.scatter(x,y)			
pl.show()
```
# 範例練習三: 散佈圖 (Scatter plot)範例二
```
import matplotlib.pylab as pl
import numpy as np

# 產生資料
x = np.random.random(100)
y = np.random.random(100)

#畫圖
#pl.scatter(x,y,s=x*500,c=u'r',marker=u'*')	
pl.scatter(x,y,s=x*500,c=u'b',marker=u'p')	
pl.show()
```
#### matplotlib使用函數:pl.scatter
```
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
有許多參數設定:請參看原始網站

s指大小，c指顏色，marker指符號形狀
```
### matplotlib.markers符號形狀
```
https://matplotlib.org/api/markers_api.html?highlight=marker
上網看看如何改變markers
```

# 範例練習四:
```
import numpy as np
import pylab as pl

# 產生資料
x = np.arange(0.0, 2.0*np.pi, 0.01)
y = np.sin(x)						
z = np.cos(x)						

#畫圖
pl.plot(x, y, label='sin()')
pl.plot(x, z, label='cos()')
pl.xlabel('x')		
pl.ylabel('y')
pl.title('sin-cos') 


pl.legend(loc='center')	
#pl.legend(loc='upper right')	
#pl.legend()				
pl.show()
```
### matplotlib.pyplot.legend
```
語法：legend(*args)
https://ithelp.ithome.com.tw/articles/10201670
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
```
### 範例練習五:帶有數學公式的圖形
```
import numpy as np
import matplotlib.pyplot as plt


# 產生資料
x = np.linspace(0, 2*np.pi, 500)
y = np.sin(x)
#z = np.cos(x*x)
z = np.cos(x*x*x)


#畫圖
plt.figure(figsize=(8,5))
#標籤前後加上$，代表以內嵌的LaTex引擎顯示為公式
plt.plot(x,y,label='$sin(x)$',color='red',linewidth=5)	#紅色，2個像素寬
plt.plot(x,z,'b--',label='$cos(x^2)$')		#b::藍色，--::虛線
plt.xlabel('Time(s)')
plt.ylabel('Volt')
plt.title('Sin and Cos figure using pyplot')
plt.ylim(-1.2,1.2)

plt.legend()							
plt.show()
```

### 範例練習六:多圖合併
```
import numpy as np
import matplotlib.pyplot as plt

# 產生資料
x= np.linspace(0, 2*np.pi, 500)	
y1 = np.sin(x)					
y2 = np.cos(x)
y3 = np.sin(x*x)

#畫圖
plt.figure(1)					#建立圖形
#create three axes
ax1 = plt.subplot(2,2,1)			#第一列第一行圖形
ax2 = plt.subplot(2,2,2)			#第一列第二行圖形
ax3 = plt.subplot(2,1,2)			#第二列

##設定第一張圖
plt.sca(ax1)					#選擇ax1
plt.plot(x,y1,color='red')		#繪製紅色曲線
plt.ylim(-1.2,1.2)				#限制y坐標軸範圍

##設定第二張圖
plt.sca(ax2)					#選擇ax2
plt.plot(x,y2,'b--')			#繪製藍色曲線
plt.ylim(-1.2,1.2)

##設定第三張圖
plt.sca(ax3)					#選擇ax3
plt.plot(x,y3,'g--')
plt.ylim(-1.2,1.2)

##多圖並呈
plt.show()
```
### 範例練習七:
```
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


# 產生資料
x,y = np.mgrid[-2:2:20j, -2:2:20j]
z = 50 * np.sin(x+y)		#測試資料

#畫圖
ax = plt.subplot(111, projection='3d')	#三維圖形
ax.plot_surface(x,y,z,rstride=2, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')			#設定坐標軸標籤
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.show()
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
