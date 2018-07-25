# 第4步：构建，培训和评估您的模型
在本节中，我们将致力于构建，培训和评估我们的模型。 在第3步中，我们选择使用n-gram模型或序列模型，使用我们的S / W比率。 现在，是时候编写我们的分类算法并对其进行训练。 我们将使用TensorFlow与tf.keras API进行此操作。

使用Keras构建机器学习模型就是将层，数据处理构建块组装在一起，就像我们组装乐高积木一样。 这些层允许我们指定要对输入执行的转换序列。 由于我们的学习算法采用单个文本输入并输出单个分类，因此我们可以使用Sequential模型API创建线性图层堆栈。
![](https://developers.google.com/machine-learning/guides/text-classification/images/LinearStackOfLayers.png)
#### 图9：线性堆叠层

根据我们是构建n-gram还是序列模型，输入层和中间层的构造将不同。 但无论模型类型如何，最后一层对于给定问题都是相同的。
#### 构建最后一层
当我们只有2个类（二进制分类）时，我们的模型应输出单个概率分数。 例如，对于给定的输入样本输出0.2意味着“该样本在0级中的20％置信度，在类1中的80％。”为了输出这样的概率分数，最后一层的激活函数应该是 sigmoid函数，用于训练模型的损失函数应该是二元交叉熵（见图10，左）。

当有超过2个类（多类分类）时，我们的模型应该为每个类输出一个概率分数。 这些分数的总和应为1.例如，输出{0：0.2,1：0.7,2：0.1}意味着“该样本在0级中的20％置信度，在1级中的70％，以及10级 为了输出这些分数，最后一层的激活函数应该是softmax，用于训练模型的损失函数应该是分类交叉熵。 （见图10，右）。
![](https://developers.google.com/machine-learning/guides/text-classification/images/LastLayer.png)
#### 图10：最后一层
下面的代码定义了一个函数，该函数将类的数量作为输入，并输出适当数量的层单元（1个单元用于二进制分类;否则每个类1个单元）和相应的激活函数：
```
def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation
 ```
以下两节介绍了n-gram模型和序列模型的剩余模型层的创建。

当S / W比率很小时，我们发现n-gram模型比序列模型表现更好。 当存在大量小的密集向量时，序列模型更好。 这是因为在密集空间中学习嵌入关系，这在许多样本中都是最好的。

## 构建n-gram模型[选项A]
我们将独立处理token（不考虑词序）的模型称为n-gram模型。 简单的多层感知器（包括逻辑回归），GBDT和支持向量机模型都属于这一类; 他们无法利用任何有关文本排序的信息。

我们比较了上面提到的一些n-gram模型的性能，并观察到多层感知器（MLP）通常比其他选项表现更好。 MLP易于定义和理解，提供良好的准确性，并且需要相对较少的计算。

以下代码定义了tf.keras中的两层MLP模型，添加了几个Dropout层用于正则化（以防止过度拟合训练样本）。
```python
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model
    ```
    
 
