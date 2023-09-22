# Stock Price Movement with Transformer Architecture
Welcome to the Stock Movement Prediction with Transformer Repository. This project offers a comprehensive codebase featuring a state-of-the-art Transformer architecture implemented in PyTorch. Its primary objective is to forecast stock movements by integrating daily Reddit news headlines with historical data from the DJIA.

# Data
The dataset comprises the top 25 daily news headlines sourced from Reddit, meticulously ranked based on user votes. Additionally, it includes comprehensive historical data pertaining to the Dow Jones Industrial Average (DJIA). This historical data encompasses essential financial metrics such as opening price, closing price, adjusted closing price, high and low prices, as well as the volume of shares traded.

Data Range: <b>2008-08-08</b> to <b>2016-07-01</b>

### Pre Processing Data
*  Initial Data Preparation
    * To prepare the news data for analysis, an initial cleaning step is performed.
    * As Reddit news is in string format, further preprocessing is essential to convert it into numerical data suitable for training the transformer model.

* Sentiment Analysis
  * Sentiment analysis of each Reddit news headline is conducted using a pre-trained NLP model known as [FinBert](https://huggingface.co/ProsusAI/finbert).
  * This analysis yields positive, negative, and neutral sentiment scores for each news item.
  * The sentiment analysis process is documented in the 'Sentiment.ipynb' notebook.
  * The resulting sentiment data is saved as a PyTorch file ('Data.pt') for future use.
