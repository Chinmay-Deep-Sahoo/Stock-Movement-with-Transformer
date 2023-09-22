# Stock Price Movement with Transformer Architecture
Welcome to the Stock Movement Prediction with Transformer Repository. This project offers a comprehensive codebase featuring a state-of-the-art Transformer architecture implemented in PyTorch. Its primary objective is to forecast stock movements by integrating daily Reddit news headlines with historical data from the DJIA.

# Data
The dataset comprises the top 25 daily news headlines sourced from Reddit, meticulously ranked based on user votes. Additionally, it includes comprehensive historical data pertaining to the Dow Jones Industrial Average (DJIA). This historical data encompasses essential financial metrics such as opening price, closing price, adjusted closing price, high and low prices, as well as the volume of shares traded.

Data Range: <b>2008-08-08</b> to <b>2016-07-01</b>

### Pre Processing Data
*  Initial Data Preparation
    * To prepare the news data for analysis, an initial cleaning step is performed to remove unessential characters and fill nan values.
    * As Reddit news is in string format, further preprocessing is essential to convert it into numerical data suitable for training the transformer model.

* Sentiment Analysis
  * Sentiment analysis of each Reddit news headline is conducted using a pre-trained NLP model known as [FinBert](https://huggingface.co/ProsusAI/finbert).
  * This analysis yields positive, negative, and neutral sentiment scores for each news item.
  * The sentiment analysis process is documented in the <b>'Sentiment.ipynb'</b> notebook.
  * The resulting sentiment data is saved as a PyTorch file ('Data.pt') for future use.

# Transformer Architecture
A transformer represents a sophisticated deep-learning model that leverages self-attention mechanisms to assign varying degrees of importance to individual segments of input data. It finds prominent application in domains such as natural language processing and other scenarios involving sequential data, as exemplified in this context by time series analysis. They are the next generation of RNN and LSTM, introduced in 2017 in the research paper, [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).
<p align = "center">
<img width="393" alt="transformers (1)" src="https://github.com/Chinmay-Deep-Sahoo/Stock-Movement-with-Transformer/assets/118956460/ba45bbcf-3e08-4645-b0b5-b829a9f76912">
</p>
The transformer consists of various blocks the details of each are provided in the resources section of this repo. The code of all the blocks is provided in <b>'Blocks.py'</b>, and the transformer model is provided in <b>'Transformer_Model.ipynb'</b> notebook.

### Self-Attention Block
Self-attention in a transformer model enables it to dynamically focus on different segments of the input sequence when making predictions. This mechanism allows for a comprehensive contextual understanding of the entire sequence while encoding individual input elements. Self-attention introduces the concepts of Keys, Queries, and Values, which the transformer model learns during its training process. To grasp the concept at a high level:
*   <b>Queries</b> can be likened to asking questions, such as, was the news positive? or, were a lot of shares traded the day before yesterday?
*   <b>Keys</b> examine the input data sequence and identify which data point best corresponds to the query.
*   <b>Values</b> dictate how the data point should be represented in a more meaningful manner.

### Multi-Head Attention Block
Multi-head attention is a concept closely related to self-attention, with a significant distinction lying in its utilization of multiple self-attention blocks, often referred to as 'heads.' These heads operate independently and, once their computations are complete, their outputs are concatenated together.
<p align = "center">
<img width="393" alt="transformers (1)" src="https://github.com/Chinmay-Deep-Sahoo/Stock-Movement-with-Transformer/assets/118956460/f760fdf4-ede6-46fd-8649-4950def105cf">
</p>


# Resources
The following resources were consulted when seeking assistance with specific sections of Transformers:
*   Transformers overview: [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY)
*   Masked Attention Layer: [Attention is all you need. A Transformer Tutorial: 7. Decoder Masked Multihead Attention](https://www.youtube.com/watch?v=SyWMFPFvsd0)
*   Positional Encoding: [Attention is all you need. A Transformer Tutorial: 5. Positional Encoding](https://www.youtube.com/watch?v=LSCsfeEELso)
### Additional resources to learn about transformers:
*   [Sequence Models - deeplearning.ai](https://www.coursera.org/learn/nlp-sequence-models?)
*   Medium article, Nikhil Verma - [Query, Key and Value in Attention mechanism](https://lih-verma.medium.com/query-key-and-value-in-attention-mechanism-3c3c6a2d4085)
