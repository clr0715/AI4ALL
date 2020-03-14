
# Numpy實戰 使用Google Colab學習Python與人工智慧
```
使用Google Colab平台學習Python程式設計
使用Google Colab平台學習Python資料分析技術
使用Google Colab平台學習機器學習(machine Learning)
使用Google Colab平台學習深度學習(deep Learning)
使用Google Colab平台學習強化學習(reinforcement Learning)
```
# Google Colab
```
https://colab.research.google.com/
```
```
Google Colab is a free cloud service and now it supports free GPU!

You can;
improve your Python programming language coding skills.
develop deep learning applications using popular libraries such as Keras,
TensorFlow, PyTorch, OpenAI gymand OpenCV.
The most important feature that distinguishes Colab from other free cloud services is;
Colab provides GPU and is totally free.
```
# 確認Google Colab上的套件使用版本
```
import pandas as pd
print("pandas version: %s" % pd.__version__)

import matplotlib
print("matplotlib version: %s" % matplotlib.__version__)

import numpy as np
print("numpy version: %s" % np.__version__)

import sklearn
print("scikit-learn version: %s" % sklearn.__version__)

import tensorflow as tf
print("tensorflow version: %s" % tf.__version__)

import torch
print("PyTorch version: %s" %torch.__version__)
print("2020年3月PyTorch version最新版本 是1.4 請參閱https://pytorch.org/")
```
## NUMPY

### NUMPY ndarray(N-Dimensional Arrays)
### NUMPY ndarray(N-Dimensional Arrays)重要屬性
```
shape
dimension
```
```
import numpy as np
ar2=np.array([[0,3,5],[2,8,7]])
# ar2.shape
ar2.ndim
```
### ndarray資料型態(dtype)與型態轉換(astype)
```
ar=np.array([2,4,6,8]); 
ar.dtype
```
### 下列程式執行後的結果為何?
```
f_ar = np.array([13,-3,8.88])
f_ar

intf_ar=f_ar.astype(int)
intf_ar
```
## 建立array(陣列)的方式與實作
```
1.使用Python內建的array()建立陣列
2.使用numpy提供的創建函數建立陣列
3.直接生成使用genfromtxt()方法建立陣列
```
## 1.使用Python內建的array()建立陣列
```
import numpy as np
x = np.array([[1,2.0],[0,0],(1+1j,3.)])
```




## 3.直接生成使用genfromtxt()方法建立陣列

```
import csv
import numpy as np

x = '''1,3,2,3,1,2,3,4
2,4,5,0.6,5,6,7,8
3,7,8,9,9,10,11,12
4,1,1.1,1.2,13,14,15,16
'''
with open("abc.txt",mode="w",encoding="utf-8") as file:
  file.write(x)
file.close()

np.genfromtxt('abc.txt', delimiter=',', invalid_raise = False)
```
## 2.使用numpy提供的創建函數建立陣列
```
eye
zeros
ones
linspace
indices
diag
tile
```
```
import numpy as np
ar9 = np.eye(3);
ar9
```

### 
```
import numpy as np
np.zeros((2, 3))
```
```
import numpy as np
np.ones((4, 7))
```
### 
```
import numpy as np
np.arange(2, 3, 0.1) # start, end, step
```
### 
```
import numpy as np
np.linspace(1., 4., 6) # start, end, num
```
###
```
import numpy as np
np.linspace(2.0, 3.0, num=5)
```
###
```
import numpy as np
np.linspace(2.0, 3.0, num=5, endpoint=False)
```
###
```
import numpy as np
np.linspace(2.0, 3.0, num=5, retstep=True)
```
###
```
import numpy as np
np.indices((3, 3))
```
```
import numpy as np
ar10=np.diag((2,1,4,6));
ar10
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, 2)
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, (2, 2))
```
```
import numpy as np
a = np.array([0, 1, 2])
np.tile(a, (2, 1, 2))
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),3)
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),(2,2))
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.array([range(i, i + 3) for i in [2, 4, 6]])
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),3)
```
```
import numpy as np
np.tile(np.array([[1,2],[6,7]]),(2,2))
```
### NUMPY ndarray 運算(Array shape manipulation)
### NUMPY ndarray 運算(Array shape manipulation)重要屬性
```
reshape
ravel()
T
newaxis
```
###
```
import numpy as np
x = np.arange(2,10).reshape(2,4)
x
```
```
import numpy as np
y = np.arange(2,10).reshape(4,2)
y
```
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)]); 
ar
```
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.ravel()
```
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.T
```
```
import numpy as np
ar=ar[:, np.newaxis]; ar.shape
ar
```
### 下列程式執行後的結果為何?
```
import numpy as np
ar=np.array([np.arange(1,6),np.arange(10,15)])
ar.T.ravel()
```
```
import numpy as np
ar=np.array([14,15,16])
ar=ar[:, np.newaxis]
ar.shape
```
## NUMPY ndarray (N-Dimensional Arrays)切片與運算
### 1.索引(index)
```
import numpy as np
x = np.arange(2,10)
# x
print(x[0])
print(x[-2])
print(x[-1])
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
ar[1,2]
```
### 2.切片運算(slice)
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[1:5:2]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[1:6:2]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[:4]
```
### 下列程式執行後的結果為何?
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[2,:] 
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[:,1]  
```
```
import numpy as np
ar = np.array([[2,3,4],[9,8,7],[11,12,13]])
# ar
ar[2,-1]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[::3]
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[:3]=1
ar
```
```
import numpy as np
ar=2*np.arange(6)
# ar
ar[2:]=np.ones(4)
ar
```
## NUMPY ndarray(N-Dimensional Arrays) Reduction Operations
```
prod()
sum()
mean()
median(ar)
```
```
import numpy as np
ar=np.arange(1,5)
ar.prod()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
ar.sum()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
ar.mean()
```
```
import numpy as np
ar=np.array([[2,3,4],[5,6,7],[8,9,10]])
np.median(ar)
```
### 下列程式執行後的結果為何?
```
ar=np.array([np.arange(1,6),np.arange(1,6)])
# ar
np.prod(ar,axis=0)
```
```
ar=np.array([np.arange(1,6),np.arange(1,6)])
# ar
np.prod(ar,axis=1)
```
## NUMPY ndarray 運算 Universal Functions:Fast Element-Wise Array Functions
```
sqrt
exp
```
```
import numpy as np
arr = np.arange(10)
np.sqrt(arr)
```
```
import numpy as np
arr = np.arange(10)
np.exp(arr)
```
## NUMPY ndarray 運算(A矩陣與B矩陣間的運算)

```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr+arr
```
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr*arr
```
```
import numpy as np
ar=np.array([[1,1],[1,1]])
ar2=np.array([[2,2],[2,2]])
ar*ar2
```
```
import numpy as np
ar=np.array([[1,1],[1,1]])
ar2=np.array([[2,2],[2,2]])
ar.dot(ar2)
```
### 下列程式執行後的結果為何?
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr-arr
```
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
1/arr
```
```
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr
arr ** 0.5
```
```
import numpy as np
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result
```
```
import numpy as np
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
result
```
## NUMPY ndarray 運算 A矩陣與B矩陣間的運算 Broadcasting(廣播機制)
```
multiply
```
```
import numpy as np
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.multiply(x1, x2)
```
```
import numpy as np
np.arange(3) + 5
```
```
import numpy as np
np.ones((3, 3)) + np.arange(3)
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.arange(3).reshape((3, 1)) + np.arange(3)
```
## 使用<Numpy random模組>產生隨機資料
```
numpy.random.randint(low, high=None, size=None, dtype='l')

seed
random.randint

```
```
import numpy as np
np.random.randint(1,5)
```
```
import numpy as np
np.random.randint(-5,5,size=(2,2))
```
```
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size=6) 
x1
```
```
import numpy as np
np.random.seed(0)
x2 = np.random.randint(10, size=(3, 4))
x2
```
```
import numpy as np
np.random.seed(0)
x3 = np.random.randint(10, size=(3, 4, 5))
x3
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.random.randint(1,size=5)
```
## NUMPY ndarray 運算(A矩陣與B矩陣間的convolute運算)
```
numpy.convolve(a, v, mode=‘full’ )

convolve
```
```
import numpy as np
np.convolve([1, 2, 3], [0, 1, 0.5])
```
```
import numpy as np
np.convolve([1,2,3],[0,1,0.5], 'same')
```
```
import numpy as np
np.convolve([1,2,3],[0,1,0.5], 'valid')
```
### 下列程式執行後的結果為何?
```
import numpy as np
np.convolve([1, 2, 3], [0, 1, 0.5])
```
## NUMPY ndarray(N-Dimensional Arrays)檔案輸入與輸出
```
https://ithelp.ithome.com.tw/articles/10196167
save()
load()
```
## NUMPY ndarray 運算 - 排序sort
```
https://github.com/femibyte/mastering_pandas/blob/master/MasteringPandas-chap3_DataStructures.ipynb

學生報告:舉例說明numpy陣列的各項排序運算

sort
```
```
import numpy as np
ar=np.array([[3,2],[10,-1]])
# ar
ar.sort(axis=1)
ar
```
### 下列程式執行後的結果為何?
```
import numpy as np
ar=np.array([[3,2],[10,-1]])
# ar
ar.sort(axis=0)
ar
```
```
import numpy as np
ar=np.random.randint(10,size=5)
ar.sort()
ar[::-1]
```

```
