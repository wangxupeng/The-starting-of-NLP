# 第3步：准备数据
在将数据提供给模型之前，需要将其转换为模型可以理解的格式。

首先，我们收集的数据样本可能按特定顺序排列。我们不希望任何与样本排序相关的信息影响文本和标签之间的关系。例如，如果数据集按类排序，然后分成训练/验证集，则这些集将不代表数据的整体分布。

确保模型不受数据顺序影响的简单最佳实践是在执行任何其他操作之前始终将数据混洗。如果您的数据已经拆分为培训和验证集，请确保转换验证数据的方式与转换培训数据的方式相同。如果您还没有单独的培训和验证集，您可以在洗牌后拆分样本;通常使用80％的样本进行培训，20％进行验证。

其次，机器学习算法将数字作为输入。这意味着我们需要将文本转换为数字向量。此过程分为两个步骤：

1.```Tokenization```：将文本分为单词或较小的子文本，这样可以很好地概括文本和标签之间的关系。 这决定了数据集的“词汇表”（数据中存在的唯一token）。

2.```Vectorization```：定义一个很好的数值测量来表征这些文本.

让我们看看如何对n-gram向量和序列向量执行这两个步骤，以及如何使用特征选择和规范化技术优化向量表示.

### N-gram载体[选项A]
在随后的段落中，我们将看到如何对n-gram模型进行tokenization和vectorization。 我们还将介绍如何使用特征选择和规范化技术优化n-gram表示。

在n-gram向量中，文本被表示为唯一n-gram的集合：n个相邻tokens的组（通常是单词）。 考虑一下文本```The mouse ran up the clock```。 在这里，单词unigrams（n = 1）是 ```['the', 'mouse', 'ran', 'up', 'clock'], the word bigrams (n = 2) are ['the mouse', 'mouse ran', 'ran up', 'up the', 'the clock']```,等等

#### Tokenization
我们发现，将单词unigrams + bigrams标记为提供良好的准确性，同时减少计算时间。

#### Vectorization
一旦我们将文本样本分成n-gram，我们需要将这些n-gram转换为我们的机器学习模型可以处理的数值向量。 下面的示例显示了为两个文本生成的unigrams和bigrams分配的索引。

```
Texts: 'The mouse ran up the clock' and 'The mouse ran down'
Index assigned for every token: {'the': 7, 'mouse': 2, 'ran': 4, 'up': 10,
  'clock': 0, 'the mouse': 9, 'mouse ran': 3, 'ran up': 6, 'up the': 11, 'the
clock': 8, 'down': 1, 'ran down': 5}
```
将索引分配给n-gram后，我们通常使用以下选项之一进行矢量化。

#### One-hot encoding:
每个示例文本都表示为一个向量，表示文本中是否存在token中。

```'The mouse ran up the clock' = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]```

#### Count encoding:
每个示例文本都表示为一个向量，指示文本中token的计数。 请注意，对应于unigram'the'（下面用粗体）对应的元素现在表示为2，因为单词“the”在文本中出现两次。
```'The mouse ran up the clock' = [1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1]```

#### Tf-idf encoding: 
上述两种方法的问题在于，在所有文档中以相似频率出现的常用词（即，对数据集中的文本样本不是特别独特的词）不会受到惩罚。 例如，像“a”这样的单词将在所有文本中非常频繁地出现。 因此，对于“the”而言，比其他更有意义的单词更高的token数量并不是非常有用。
```
'The mouse ran up the clock' = [0.33, 0, 0.23, 0.23, 0.23, 0, 0.33, 0.47, 0.33,
0.23, 0.33, 0.33] (See Scikit-learn TdidfTransformer)
```
还有许多其他矢量表示，但以上三种是最常用的。

我们观察到tf-idf编码在准确性方面略优于其他两个（平均：高出0.25-15％），并建议使用此方法对n-gram进行矢量化。 但是，请记住它占用更多内存（因为它使用浮点表示）并且需要更多时间来计算，特别是对于大型数据集（在某些情况下可能需要两倍的时间）。

#### 特征选择
当我们将数据集中的所有文本转换为单词uni + bigram标记时，我们最终可能会有数万个标记。 并非所有这些令牌/特征都有助于标签预测。 因此，我们可以删除某些令牌，例如在数据集中极少发生的令牌。 我们还可以测量特征重要性（每个标记对标签预测的贡献程度），并且仅包括信息量最大的标记。

有许多统计函数可以获取特征和相应的标签并输出特征重要性分数。 两个常用的函数是f_classif和chi2。 我们的实验表明，这两个功能同样表现良好。

更重要的是，我们发现许多数据集的精度达到了大约20,000个特征（见图6）。 在此阈值上添加更多功能的贡献非常小，有时甚至会导致过度拟合并降低性能。
```
我们在这里的测试中只使用英文文本。 理想的功能数量可能因语言而异; 这可以在后续分析中探讨。
```
![](https://developers.google.com/machine-learning/guides/text-classification/images/TopKvsAccuracy.svg)
#### 图6：前K特征与精度。 在整个数据集中，精度平稳在20K特征左右。
### 正常化
标准化将所有要素/样本值转换为小值和类似值。 这简化了学习算法中的梯度下降收敛。 从我们所看到的情况来看，数据预处理期间的规范化似乎并没有在文本分类问题中增加太多价值; 我们建议您跳过此步骤。

以下代码汇总了上述所有步骤：

*  将文本样本标记为单词uni + bigrams，
*  使用tf-idf编码进行矢量化，
*  通过丢弃出现少于2次的标记并使用f_classif计算要素重要性，仅从标记向量中选择前20,000个要素。
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val
```  
使用n-gram向量表示，我们丢弃了大量关于单词顺序和语法的信息（当n> 1时，我们可以保留一些部分排序信息）。 这被称为词袋方法。 该表示与不考虑排序的模型结合使用，例如逻辑回归，多层感知器，GBDT，支持向量机。

### 序列载体[选项B]
在随后的段落中，我们将看到如何对序列模型进行标记化和矢量化。我们还将介绍如何使用特征选择和规范化技术优化序列表示。

对于某些文本示例，单词顺序对于文本的含义至关重要。例如，句子，“我曾经恨我的通勤。我的新自行车完全改变了“只有在按顺序阅读时才能理解。诸如CNN / RNN之类的模型可以从样本中的单词顺序推断出含义。对于这些模型，我们将文本表示为一系列标记，保留顺序。

#### Tokenization
文本可以表示为字符序列或单词序列。我们发现使用单词级表示比字符标记提供更好的性能。这也是行业遵循的一般规范。只有当文本有很多拼写错误时才使用字符标记，这通常不是这种情况。

#### Vectorization
一旦我们将文本样本转换为单词序列，我们需要将这些序列转换为数字向量。下面的示例显示分配给为两个文本生成的unigrams的索引，然后显示转换第一个文本的token索引序列。
```
Texts: 'The mouse ran up the clock' and 'The mouse ran down'
Index assigned for every token: {'clock': 5, 'ran': 3, 'up': 4, 'down': 6, 'the': 1, 'mouse': 2}.
注意：'the'最常出现，因此索引值为1。
有些库为未知token保留索引0，就像这里的情况一样。
token索引序列'The mouse ran up the clock' = [1, 2, 3, 4, 1, 5]
````
有两个选项可用于矢量化标记序列：
One-hot encoding: 在n维空间中使用单词向量表示序列，其中n =词汇量的大小。 当我们将字符标记为字符时，这种表示非常有效，因此词汇量很小。 当我们将其标记为单词时，词汇表通常会有数万个token，使得one-hot encoding的向量非常稀疏且效率低下。 例：
```
'The mouse ran up the clock' = [
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0],
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0]
]
```
Word embeddings: 单词具有与之相关的含义。 结果，我们可以在密集的向量空间（〜几百个实数）中表示单词标记，其中单词之间的位置和距离表示它们在语义上有多相似（参见图7）。 这种表示称为单词嵌入。
![](https://developers.google.com/machine-learning/guides/text-classification/images/WordEmbeddings.png)
#### 图7：Word嵌入
序列模型通常具有这样的嵌入层作为它们的第一层。 该层学习在训练过程中将单词索引序列转换为单词嵌入向量，使得每个单词索引被映射到表示该单词在语义空间中的位置的实数值的密集向量（参见图8）。
![](https://developers.google.com/machine-learning/guides/text-classification/images/EmbeddingLayer.png)
#### 图8：嵌入层

#### 特征选择
并非我们数据中的所有单词都有助于标签预测。 我们可以通过从词汇表中丢弃罕见或不相关的单词来优化我们的学习过程。 事实上，我们观察到使用最常见的20,000个特征通常就足够了。 对于n-gram模型也是如此（参见图6）。

让我们将所有上述步骤放在序列矢量化中。 以下代码执行以下任务：

*  将文本标记为单词
*  使用前20,000个token创建词汇表
*  将标记转换为序列向量
*  将序列填充到固定的序列长度
```python
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index
```    
#### 标签矢量化
我们看到了如何将示例文本数据转换为数字向量。 必须对标签应用类似的过程。 我们可以简单地将标签转换为范围[0，num_classes - 1]中的值。 例如，如果有3个类，我们可以使用值0,1和2来表示它们。 在内部，网络将使用one-hot encoding来表示这些值（以避免推断标签之间的错误关系）。 这种表示取决于我们在神经网络中使用的损失函数和最后一层激活函数。 我们将在下一节中详细了解这些内容。
