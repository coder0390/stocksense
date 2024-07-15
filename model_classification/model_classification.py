import os

from torch import nn

HIDDEN_DIM = hidden_dim = 250
NUM_CLASSES = num_classes = 2
MAX_LENGTH = 300
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
EPS = 1e-8
DROPOUT = 0.2
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)  # 添加Dropout层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT输出维度：(batch_size, seq_length, hidden_size)
        out, _ = self.bilstm(bert_output.last_hidden_state)
        # Bi-LSTM输出维度：(batch_size, seq_length, hidden_dim * 2)
        out = self.dropout(out[:, -1, :])  # 应用Dropout层
        out = self.fc(out)  # 取最后一个时间步的输出
        # 全连接层输出维度：(batch_size, num_classes)
        return out

def predict(model, tokenizer, text):
    # 将文本转换成模型所需的输入格式
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # 使用模型进行预测
    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask'])
        _, prediction = torch.max(output, dim=1)

    return prediction.item()



import torch
from transformers import BertTokenizer, BertModel

# # 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 模型文件路径
model_path = 'BERT-BiLSTM_sentiment_analysis.pt'
m_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model = SentimentClassifier(bert_model, HIDDEN_DIM, NUM_CLASSES).to('cpu')
model.load_state_dict(m_state_dict)

import pandas as pd
# from your_database_module import connect_to_database, fetch_news_by_id

def analyze_sentiment_for_news(news_id):
    # 连接数据库
    db = connect_to_database()

    # 从数据库获取特定ID的新闻内容
    news_content = fetch_news_by_id(db, news_id)

    # 进行情感分析预测
    prediction = predict(model, tokenizer, news_content)

    # 根据预测输出结果
    if prediction == 1:
        sentiment = 'positive'
    elif prediction == 0:
        sentiment = 'negative'
    else:
        sentiment = 'undefined'

    print(f"Sentiment prediction for news ID {news_id}: {sentiment}")

# 示例调用：假设要分析ID为1的新闻
analyze_sentiment_for_news(1)


# predict all from input
# import pandas as pd

# # 读取CSV文件
# input_csv = 'input.csv'
# df = pd.read_csv(input_csv)

# df.dropna(subset=['content'], inplace=True)

# # 对content列进行情感分析并显示进度
# predictions = []
# total_rows = len(df)
# for index, row in df.iterrows():
#     prediction = predict(model, tokenizer, row['content'])
#     predictions.append(prediction)
#     print(f"Processed {index + 1}/{total_rows} rows")

# # 将预测结果添加到DataFrame中
# df['predict'] = predictions

# # 将结果保存到新的CSV文件
# output_csv = 'output.csv'
# df.to_csv(output_csv, index=False)

# print(f"Predictions saved to {output_csv}")