---
layout: post
title:  "Turn your raw sales numbers into deep sales insights and intelligent forecasts"
author: weisseo
categories: [forecasting]
image: assets/images/forecast_tree.png
mathjax: true
---

In the last years technologies like data science, artificial intelligence and predictive analytics have become 
increasingly popular. There are several applications of these technologies which include target marketing, churn 
prediction or sales prediction.

The academic field behind these technologies is called “machine learning”.
The core idea is that the computer (“machine”) learns autonomously from data and generates business insights and leads 
to data-driven decision making.

To illustrate and demystify the technology, in this article we will walk through the example of sales prediction. We 
will cover how to get from raw data to valuable predictions with machine learning. The article can be understood by a 
nontechnical reader.

On the platform “Kaggle.com” companies can upload their data and let data scientist compete to build the best machine 
learning algorithm for their problem. For example, the German store Rossman outsourced their sales prediction in 2015.
Rossmann store managers were tasked with predicting their daily sales for up to six weeks in advance. Predicting the 
sales of store items helps planning how much products have to be kept in the warehouse. Store sales are influenced by
many factors, including promotions, competition, school and state holidays, seasonality, and locality. And if it is 
about predicting from data, machine learning is definitely the right technology.

For reasons of simplicity, we will not go through the Rossmann use case, but will instead work with another Kaggle
dataset, which was provided by one of the largest Russian software firms - 1C Company. In contrast to the Rossmann 
challenge, we do not want to predict daily sales, but monthly sales instead.

We have 33 months of sales data and want to predict the sales for the next month. Let us have a look at what we want to
predict:

shop_id | item_id | sales
--- | --- | ---
5 | 5037 | ?
5 | 5320 | ?
5 | 5233 | ?
5 | 5232 | ?
5 | 5268 | ?

Given the shop id and a particular item we want to predict how many items will be sold. Let us now have a look on a 
snippet of how our past sales data looks like:

Table

Here, one row in the table is one transaction on a particular day. We also have the additional columns “date_block_num” 
which enumerates the months and “item price”. “item_cnt_day” stands for how many items have been sold with this 
particular transaction. It can also be a negative value if an item was returned.
At the moment, the data is not yet in the form, we want it to have. We want to predict the monthly sales for a
particular item and shop. So, we have to count all the items that have been sold in a month and sum up the 
“item_cnt_day”. In data science jargon these transformations are called pivot tables or “groupbys”. It is part of the 
“data wrangling process”: Transforming the raw data into the desired form. This part can take most of the data 
scientists time. After the transformation our data table looks like that:

Table

For example, in June 2013, the item “482” was sold 2 times in shop 2. Let us now try to predict the sales of the next
month without any machine learning. We will use the sales of the last month to predict the sales of this month. We have 
to add the column “item_cnt_last_month” for that. So, for the row June 2013, shop 2, item “482”, we have to search for 
the entry May 2013, shop 2, item “482” and copy the data from “item_cnt_mont”. This is another example of 
“data wrangling”. Now our table looks like this:

Table

Now we have the variable “item_cnt_month” that we want to predict in one column and our prediction “item_cnt_last_month”
in the column next to it. We can see that our prediction is not 100% accurate, but we are also not that far off. How do 
we evaluate how good our predictions are? We have to calculate the so-called scoring or loss function. One possibility 
would be to just calculate the difference between our prediction and the true sales and take the mean. For our five data
points, we are four times off by 1 and one time off by 2. So,

$$L = \frac{4 \cdot 1 + 1}{5} = 1.2$$

In practice we use the RMSE (root mean squared error). Instead of using the absolute deviation as above, we square the 
deviation and take the square root at the end. This approach penalizes large deviations more. In our example this gives 
us

formula

Now we know how good our predictions are. The lower the RMSE, the better. There are some commonly used error functions 
(also called “loss function”) like RMSE, but in practice it can be anything. For example, it could be how much money you
make if you can make a connection between the data science problem and your finances. In the data science practice, the 
error function is tailored to the specific use case. In the example above, we used only five data points. For the 
complete last month that we have the true sales data for, we obtain a RMSE of

This was all done with simple data manipulation. No machine learning was included so far. Now let us get to the next 
step. We will not only use the item counts from last month but also from two months before that. We do a similar data 
transformation as above and have these two additional features in our data table:

Table

We could now take the mean of the item counts of the last three months as a prediction. But is the last month sales not 
more important than the sales three months ago? But how much more important? And could we not use the information from 
the features “year” and “month”? Maybe there is some seasonality effect and people buy more products in summer. Here is 
the good news: machine learning will do all that for us! It will learn from the data how to use the information in the 
features.

There are several different machine learning methods like support vector machines, neural networks or decision trees. 
Here, we will introduce decision trees because they are very intuitive and easy to understand.

Image

Have a look at the decision tree that was created of one month of training data. Let us take the first row in our table 
and see what our prediction would be. The item count last month was 0. Since this is smaller than 2.5, we take the left 
path down. The item count two months ago was also 0. Again, we go down the left path and we predict the value 0.3. This 
is a very simple decision tree with only two levels. In practice the trees are a lot larger. If we train a larger 
decision tree, we obtain a RMSE of . This is better than just predicting the last month’s item count.

The decision tree is created in that way that the error function (RMSE) is minimized. It is important to note that we 
cannot test our model on the same data that our decision tree was trained on. Our decision tree was created with data 
from month 1-4 in that way that the RMSE is small on these particular months. The RMSE error that really counts is the 
RMSE on month 5 that the algorithm has not seen yet. Otherwise, we could build an extremely large tree that would 
perfectly predict the sales on month 1-4 but would be useless on month 5 because in interpreted too much into the data 
of month 1-4. In machine learning, this is called “overfitting”. It is comparable to memorizing all the answers of last 
year’s exam in university or high school. If this year’s exam is the same as last year’s than your answers would be 
perfect. But this is rarely the case in reality. The exams differ slightly from year to year, so it is better to really 
understand the material so that you generalize well.

The decision tree is very intuitive and also has the advantage that we can get an insight into which feature is the most
important. The higher up the feature is, the more important it is. In our case, this means that the item count last 
month is more important than the item count two months ago. Not all machine learning algorithms are that easy to 
understand. Fortunately, there are other methods which show us how important a feature is. In the next table you see the
so-called permutation importance. It shows us how important a feature is for our prediction. As expected, the sales from
last months are the most important feature. The item counts from the months before that and the item id and shop id 
also play a role. Surprisingly, the year and month are irrelevant for our prediction. Seasonality seems to not have any
influence on the sales. That is also a valuable insight.
  
Image

This is, of course, a simplified version. In the Rossmann store challenge a lot more variables were given in the data. 
We could also try to use information outside the data set like how many state holidays the given month had or what 
weather it was.