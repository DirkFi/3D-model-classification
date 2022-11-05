## 基于无监督域适应的三维模型分类研究

本方法集中于把源域（ImageNet）和目标域（3D模型渲染图）的域差异解决，从而给出对3D模型本身的正确分类。


## 跑通代码的步骤

### 环境安装
具体的环境安装放在同文件夹下的requirements.txt里；关于torch的安装，最好使用GPU版本，查看官网的安装方式：https://pytorch.org/.

### 修改数据集链接
具体的修改位置，在代码args里的test_path, train_path, val_path里，将内容改为自己的数据集即可。注意数据集格式为ImageNet类型。即

------------>文件夹
-------->类名1
-->图片1
...
-->图片n

------->类名m
-->图片...

### 运行代码
接下来就可以跑代码了，但要注意最后生成的准确率是按照多视图12张融合生成的模型准确率，并不是每一张图片一
一检测得到的准确率。 

运行的文件是res_train_main_D.py
命令行运行：
```
python res_train_main_D.py
```

# Done.