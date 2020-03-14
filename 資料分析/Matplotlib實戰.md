# Data Visualization資料視覺化
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


```

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
# 延伸閱讀:推薦的教科書

```

```
