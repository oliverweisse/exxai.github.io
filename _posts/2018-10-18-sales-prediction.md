---
layout: post
title:  "Turn your raw sales numbers into deep sales insights and intelligent forecasts"
author: weisseo
categories: [forecasting]
image: assets/images/forecast_tree.png
mathjax: true
---

In the last years technologies like data science, artificial intelligence and predictive analytics
have become increasingly popular. There are several applications of these technologies which include target
marketing, churn prediction or sales prediction.

The academic field behind these technologies is called “Machine Learning”.
The core idea is that the computer (“machine”) learns autonomously from data and generates business
insights and leads to data-driven decision making.

To illustrate and demystify the technology, in this article we will walk through the example of sales prediction.
We will cover how to get from raw data to valuable predictions with machine learning.
The article can be understood by a nontechnical reader.

On the platform “Kaggle.com” companies can upload their data and let data scientist compete
to build the best machine learning algorithm for their problem.
For example, the German store Rossman outsourced their sales prediction in 2015.
Rossmann store managers were tasked with predicting their daily sales for up to six weeks in advance.
Predicting the sales of store items helps planning how much products have to be kept in the warehouse.
Store sales are influenced by many factors, including promotions,
competition, school and state holidays, seasonality, and locality.
And if it is about predicting from data, machine learning is definitely the right technology.

For reasons of simplicity, we will not go through the Rossmann use case, but will instead work
with another Kaggle dataset, which was provided by one of the largest Russian software firms - 1C Company.
In contrast to the Rossmann challenge, we do not want to predict daily sales, but monthly sales instead.

We have 33 months of sales data and want to predict the sales for the next month.
Let us have a look at what we want to predict:

shop_id | item_id | sales
--- | --- | ---
5 | 5037 | ?
5 | 5320 | ?
5 | 5233 | ?
5 | 5232 | ?
5 | 5268 | ?

...

$$L = \frac{4 \cdot 1 + 1}{5} = 1.2$$
