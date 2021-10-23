import argparse


def args_initialization():
    parser = argparse.ArgumentParser(description="MovieLens 1M")
    # 基本参数
    parser.add_argument("-path", type=str, default="data/", help="数据位置")
    parser.add_argument("-max-length", type=int, default=16, help="电影标题文本的最大长度")
    parser.add_argument("-batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("-embedding-dim", type=int, default=32, help="序号等信息的 Embedding 维度")
    parser.add_argument("-hidden-dim", type=int, default=64, help="隐藏层维度")
    parser.add_argument("-lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("-epochs", type=int, default=3, help="Epoch 数")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout 率")
    parser.add_argument("-seed", type=int, default=2021, help="随机种子")
    parser.add_argument("-k", type=int, default=10, help="SimRank 迭代轮数")
    parser.add_argument("-c", type=int, default=10, help="SimRank 阻尼系数")
    # TextCNN 参数
    parser.add_argument("-filter-sizes", type=list, default=[1, 3, 5],
                        help="与卷积核大小相关，每层filter的卷积核大小为(filter_size,embedding_dim)")
    # Embedding 参数
    parser.add_argument("-vocabulary_size", type=int, default=5216, help="电影文本单词数")
    parser.add_argument("-text-embedding-dim", type=int, default=300, help="电影信息文本 Embedding 维度")
    parser.add_argument("-static", type=bool, default=False, help="电影标题文本是否使用静态词向量")
    parser.add_argument("-embedding-path", type=str, default=None, help="静态词向量位置")
    # 设备
    parser.add_argument("-device", type=str, default='cuda', help="设备")
    # 推荐用户或电影的数量
    parser.add_argument("-recommend-num", type=int, default=5)
    args = parser.parse_args(args=[])
    return args
