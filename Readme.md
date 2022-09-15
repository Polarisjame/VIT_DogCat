# VIT_DogCat
用VIT尝试了下kaggle数据集的猫狗分类

## 数据集下载
> [kaggle猫狗分类](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)

## 环境配置
* `python3.7`
* `torch1.12.0+cu116`


## Quick Start:
```python
# Vit
python main.py
#use -h to get help
```

## model:
+ Vit(Vision Transformer)
+ 图像分隔拍成序列转化成transformer需要的序列数据

**大概流程:**
1. `PatchEmbedding`: input: `[batch,channels,h,w]` -> output:`[batch,num+1,emb_size]` `num`:拆分平铺后数量  +1:加一个`CLS` 其实就是经过conv后将图像拆分堆叠加上CLS_Token和Positional_Embedding
2. `Encoder`: 里面有`depth`层`EncoderLayer` 就是transformer同款，懂得都懂

> 别太期待结果，VIT在大规模数据上表现好，在这里就一言难尽li
