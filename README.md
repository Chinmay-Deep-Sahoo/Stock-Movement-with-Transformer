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
The transformer consists of various blocks the details of each are provided in the resources section of this repo. The code of all the blocks is provided in 'Blocks.py', and the transformer model is provided in 'Transformer_Model.ipynb' notebook.
