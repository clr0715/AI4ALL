#
```
走向TensorFlow 2.0：深度学习应用编程快速入门
赵英俊 (作者)　 葛娜 (责任编辑)
书　　号：978-7-121-37646-7
出版日期：2019-11-08
页　　数：180

http://www.broadview.com.cn/book/6027
```
```
第1章　Python基礎程式設計入門 1
1.1　Python的歷史 1
1.1.1　Python版本的演進 1
1.1.2　Python的工程應用情況 2
1.2　Python的基底資料型別 2
1.3　Python資料處理工具之Pandas 6
1.3.1　資料讀取和存儲 7
1.3.2　資料查看和選取 8
1.3.3　資料處理 11
1.4　Python影像處理工具之PIL 14
1.4.1　PIL簡介 14
1.4.2　PIL介面詳解 14
1.4.3　PIL影像處理實踐 18
第2章　TensorFlow 2.0快速入門 21
2.1　TensorFlow 2.0簡介 21
2.2　TensorFlow 2.0環境搭建 22
2.2.1　CPU環境搭建 22
2.2.2　基於Docker的GPU環境搭建 23
2.3　TensorFlow 2.0基礎知識 25
2.3.1　TensorFlow 2.0 Eager模式簡介 25
2.3.2　TensorFlow 2.0 AutoGraph簡介 26
2.3.3　TensorFlow 2.0低階API基礎程式設計 26
2.4　TensorFlow 2.0高階API（tf.keras） 32
2.4.1　tf.keras高階API概覽 32
2.4.2　tf.keras高階API程式設計 34
第3章　基於CNN的圖像識別應用程式設計實踐 36
3.1　CNN相關基礎理論 36
3.1.1　卷積神經網路概述 36
3.1.2　卷積神經網路結構 36
3.1.3　卷積神經網路三大核心概念 38
3.2　TensorFlow 2.0 API詳解 38
3.2.1　tf.keras.Sequential 39
3.2.2　tf.keras.layers.Conv2D 41
3.2.3　tf.keras.layers.MaxPool2D 42
3.2.4　tf.keras.layers.Flatten與tf.keras.layer.Dense 42
3.2.5　tf.keras.layers.Dropout 43
3.2.6　tf.keras.optimizers.Adam 43
3.3　專案工程結構設計 44
3.4　項目實現代碼詳解 44
3.4.1　工具類實現 45
3.4.2　cnnModel實現 46
3.4.3　執行器實現 48
3.4.4　Web應用實現 52
第4章　基於Seq2Seq的中文聊天機器人程式設計實踐 55
4.1　NLP基礎理論知識 55
4.1.1　語言模型 55
4.1.2　迴圈神經網路 57
4.1.3　Seq2Seq模型 59
4.2　TensorFlow 2.0 API詳解 61
4.2.1　tf.keras.preprocessing.text.Tokenizer 61
4.2.2　tf.keras.preprocessing.sequence.pad_sequences 62
4.2.3　tf.data.Dataset.from_tensor_slices 63
4.2.4　tf.keras.layers.Embedding 63
4.2.5　tf.keras.layers.GRU 63
4.2.6　tf.keras.layers.Dense 65
4.2.7　tf.expand_dims 65
4.2.8　tf.keras.optimizers.Adam 65
4.2.9　tf.keras.losses.SparseCategoricalCrossentropy 66
4.2.10　tf.math.logical_not 66
4.2.11　tf.concat 66
4.2.12　tf.bitcast 67
4.3　專案工程結構設計 67
4.4　項目實現代碼詳解 68
4.4.1　工具類實現 68
4.4.2　data_util實現 69
4.4.3　seq2seqModel實現 71
4.4.4　執行器實現 77
4.4.5　Web應用實現 83
第5章　基於CycleGAN的圖像風格遷移應用程式設計實踐 85
5.1　GAN基礎理論 85
5.1.1　GAN的基本思想 85
5.1.2　GAN的基本工作機制 86
5.1.3　GAN的常見變種及應用場景 86
5.2　CycleGAN的演算法原理 88
5.3　TensorFlow 2.0 API詳解 88
5.3.1　tf.keras.Sequential 88
5.3.2　tf.keras.Input 91
5.3.3　tf.keras.layers.BatchNormalization 91
5.3.4　tf.keras.layers.Dropout 92
5.3.5　tf.keras.layers.Concatenate 93
5.3.6　tf.keras.layers.LeakyReLU 93
5.3.7　tf.keras.layers.UpSampling2D 93
5.3.8　tf.keras.layers.Conv2D 93
5.3.9　tf.optimizers.Adam 94
5.4　專案工程結構設計 95
5.5　項目實現代碼詳解 96
5.5.1　工具類實現 96
5.5.2　CycleganModel實現 100
5.5.3　執行器實現 105
5.5.4　Web應用實現 109
第6章　基於Transformer的文本情感分析程式設計實踐 111
6.1　Transformer相關理論知識 111
6.1.1　Transformer基本結構 111
6.1.2　注意力機制 112
6.1.3　位置編碼 116
6.2　TensorFlow 2.0 API詳解 117
6.2.1　tf.keras.preprocessing.text.Tokenizer 117
6.2.2　tf.keras.preprocessing.sequence.pad_sequences 118
6.2.3　tf.data.Dataset.from_tensor_slices 118
6.2.4　tf.keras.layers.Embedding 118
6.2.5　tf.keras.layers.Dense 119
6.2.6　tf.keras.optimizers.Adam 119
6.2.7　tf.optimizers.schedules.LearningRateSchedule 120
6.2.8　tf.keras.layers.Conv1D 120
6.2.9　tf.nn.moments 121
6.3　專案工程結構設計 121
6.4　項目實現代碼詳解 122
6.4.1　工具類實現 122
6.4.2　data_util實現 124
6.4.3　textClassiferMode實現 128
6.4.4　執行器實現 138
6.4.5　Web應用實現 142
第7章　基於TensorFlow Serving的模型部署實踐 144
7.1　TensorFlow Serving框架簡介 144
7.1.1　Servable 145
7.1.2　Source 145
7.1.3　Loader 145
7.1.4　Manager 145
7.2　TensorFlow Serving環境搭建 146
7.2.1　基於Docker搭建TensorFlow Serving環境 146
7.2.2　基於Ubuntu 16.04搭建TensorFlow Serving環境 146
7.3　API詳解 147
7.3.1　tf.keras.models.load_model 147
7.3.2　tf.keras.experimental.export_saved_model 147
7.3.3　tf.keras.backend.set_learning_phase 148
7.4　專案工程結構設計 148
7.5　項目實現代碼詳解 149
7.5.1　工具類實現 149
7.5.2　模型檔匯出模組實現 150
7.5.3　模型檔部署模組實現 150
7.5.4　Web應用模組實現 152

```
