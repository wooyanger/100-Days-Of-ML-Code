## 1、 导入需要的库
我们每次都会导入两个必不可少的库。Numpy 是一个包含数学函数的库，Pandas 是一个用来导入和管理数据集合的库。

## 2、导入数据集合
数据集合通常是一个 csv 格式，一个CSV 文件以纯文本的方式存储表格数据，文件的每一行对应这一条数据记录。我们使用 pandas 库中的 read_csv 方法来读取一个本地的 CSV 文件并作为一个数据源。我们从数据文件中分离出独立变量和因变量的矩阵和向量。

## 3、处理丢失的数据
我们获得的数据的结构很少是一致的。数据可能由于各种原因而丢失，为了保证机器学习模型的性能，我们需要对丢失的数据进行处理。我们可以使用每列的平均值或中值来替换丢失的数据。我们使用 sklearn.preprocessing 中的 Imputer 类来完成这项任务。

## 4、编码分类数据
分类数据是包含标签值而不是数值的变量。可能的值的范围通常是一个固定的集合。例如“是”和“否”不能用在数学方程式模型中，所以我们需要将这些变量编码成数字。为此，我们从 sklearn.preprocessing 库中导入 LabelEncoder 类。

## 5、将数据集合拆分为测试集合和训练集合
我们把数据集合做成两个分区，一个叫做训练集合用来训练模型，另一个叫做测试集合来测试模型的性能。一般情况下，我们采用 80:20 的分离办法。我们使用 sklearn.model_selection 库中的 train_test_split 方法。

## 6、特征缩放
大多数机器学习算法在其计算中使用两个数据点之间的欧几里德距离，在幅度上高度变化的特征在距离计算中将比具有低幅度的特征更重。通过特征标准化或Z分数归一化。我们使用 sklearn.preprocessing 中的 StandardScalar。


## Step 1: 导入需要的库
```python
import numpy as np
import pandas as pd
```

## Step 2: 导入数据集合
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
```

## Step 3: 处理丢失的数据
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```

## Step 4: 编码分类数据
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```

### 创建虚拟变量
```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```

## Step 5: 拆分数据集到训练集和测试集中
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

## Step 6: 特征缩放
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```