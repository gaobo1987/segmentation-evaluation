### 如何对分词器做评估 Instructions for evaluation

##### 安装 How to install
先安装 segmt_eval 库，其whl文件可以在[Tulips OneBox](https://onebox.huawei.com/#teamspaceFile/1/19/4789876) 
下载。

Go to our [Tulips OneBox](https://onebox.huawei.com/#teamspaceFile/1/19/4789876) 
and download the latest installation .whl file.
```shell script
pip install segmt_eval-x.x.x-py3-none-any.whl
```

##### 数据格式 Data format
segmt_eval 依赖以下JSON列表数据格式实施评估：每一个项目包括被分词的语句 “query”，
分词后的黄金标准 “gold” 和 预测的分词 “pred”。参考下面的例子。
[eval.json](example_data/eval.json) 提供一个完整的荷兰语分词的例子。
更多例子可以在 [Tulips OneBox “Datasets”](https://onebox.huawei.com/#teamspaceFile/1/4/4789876) 文件夹中找到。

An good example is the 
[eval.json](example_data/eval.json), 
of which a snippet is shown below. It is a list of json objects 
that contain manually curated queries, and their "gold" and "pred" segmentations.
"gold" refers to the ground-truth, and "pred" refers to model-predictions. 
The evaluation functions rely on this format to perform evaluation. Such evaluation datasets in various languages 
can be found in our [Tulips OneBox "Datasets" folder](https://onebox.huawei.com/#teamspaceFile/1/4/4789876).


```json
[{
      "query":"hoofdstad van Suriname",
      "gold":[
         {
            "item":"hoofdstad",
            "lemma":"hoofdstad",
            "pos":"NOUN",
            "startOffSet":0,
            "endOffSet":9,
            "ner":"",
            "isMinimumToken":true
         },
         {
            "item":"van",
            "lemma":"van",
            "pos":"ADP",
            "startOffSet":10,
            "endOffSet":13,
            "ner":"",
            "isMinimumToken":true
         },
         {
            "item":"Suriname",
            "lemma":"suriname",
            "pos":"NOUN",
            "startOffSet":14,
            "endOffSet":22,
            "ner":[
               {
                  "ner":"LOC"
               }
            ],
            "isMinimumToken":true
         }
      ],
      "pred":[
         {
            "item":"hoofdstad",
            "lemma":"hoofdstad",
            "pos":"NOUN",
            "startOffSet":0,
            "endOffSet":9,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":false
         },
         {
            "item":"van",
            "lemma":"van",
            "pos":"ADP",
            "startOffSet":10,
            "endOffSet":13,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":true
         },
         {
            "item":"Suriname",
            "lemma":"suriname",
            "pos":"NOUN",
            "startOffSet":14,
            "endOffSet":22,
            "ner":"",
            "isMinimumToken":true,
            "isStopWord":false
         }
      ]
}]
```


##### 如何运行 How to run
准备好数据后，按照下面的代码对分词效果进行评估。目前支持三种评估：pos，lemma，和 ner。

Import the evaluate function and point it to the preprocessed evaluation dataset.
The 'key' parameter indicates which aspect of the model you want to evaluate, 
currently supported values are: "pos", "lemma" and "ner".

```python
from segmt_eval import evaluate
output = evaluate('path/to/eval/data.json', key='pos')
```

如果key是pos或者lemma，输出结果包含平均和加权的F,P,R分数以及Accuracy。 如果key是pos，输出结果进一步包含每个POS标记的F,P,R分数。

If the key is 'pos' or 'lemma', the output contains 
average and weighted F,P,R scores, and accuracy. When the key is 'pos', the result further contains F,P,R scores per individual POS category.

```json
{
  "weighted_f1": 0.883,
  "weighted_precision": 0.894,
  "weighted_recall": 0.883,
  "average_f1": 0.746,
  "average_precision": 0.736,
  "average_recall": 0.747,
  "accuracy": 0.859
}
```

如果key是ner，返回的结果是一个二元组，第一个项目包含总体F,P,R的的四种模式下的评分，
第二个项目包含基于各实体种类的F,P,R的四种模式下的评分。
我们可以根据一下四种模式的 F1, Precision, Recall 来测评NER模型表现：

* strict： 实体边界和种类严格匹配
* exact： 实体边界严格匹配, 但种类可不同
* ent_type： 实体边界和种类可非严格匹配, 但必须有重叠部分
* partial: 实体边界有重叠部分, 种类可不匹配


If the key is 'ner', 
the returned result is a tuple, the first item of which contains 
the overall F,P,R scores in the aforementioned four modes, and 
the second item contains the scores per entity label in four modes. 
And the four modes are as follows:

* strict: exact boundary surface string match and entity type
* exact: exact boundary match over the surface string, regardless of the type
* ent_type: some overlap between the system tagged entity and the gold annotation is required
* partial: partial boundary match over the surface string, regardless of the type


------

如果嫌构建评估数据JSON格式麻烦，目前也可以如下方式对NER模型进行直接评估。

You could also directly evaluate your NER model 
without composing the formatted JSON file.

```python
from segmt_eval import evaluate_ner
gold1 = ['O', 'O', 'B-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O']
pred1 = ['O', 'O', 'B-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O']
gold2 = ['O', 'O', 'O', 'O', 'O', 'O']
pred2 = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O']
gold3 = ['O', 'B-MISC', 'I-MISC', 'O', 'O', 'O']
pred3 = ['O', 'O', 'O', 'O', 'O', 'O']
gold_labels = [gold1, gold2, gold3]
pred_labels = [pred1, pred2, pred3]
label_set = ['PER', 'LOC', 'ORG', 'MISC']
results, results_per_label = evaluate_ner(gold_labels, pred_labels, label_set)
```
