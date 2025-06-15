---
title:  "Paris Airbnb visualization"
toc: true
toc_sticky: true
search: false
categories: 
  - visualization
last_modified_at: 2023-03-16T08:06:00-05:00
---

*A visualization project on Paris Airbnb market. All the plots were generated using Python.*

---

Authors: [Zixuan Tang](https://www.linkedin.com/in/zixuantang/), [Ke Chen](https://www.linkedin.com/in/kechenkc/)

Published: March, 2023

---

Hey there! Welcome to our Airbnb analysis project. We hope you find the following information informative and interesting!!

Our dataset was obtained from [**Kaggle**](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews/code?datasetId=1530930) and consists of information on Airbnb accommodations across 10 major cities, including 5 million reviews spanning from November 16th, 2008, to March 1st, 2021.

In this project, our focus was on analyzing Airbnb accommodations in Paris, which comprises <span class="color-blue">**37,907 listings and 972,309 reviews**</span>. Our analysis uncovered some fascinating insights, which we are delighted to share with you.

This page is for visualization. For machine learning models and predictions, please visit:

[Paris Airbnb predictions](https://zixtang.github.io/machine%20learning/Paris-Airbnb-predictions/)

---

# Where are the accommodations?

Let's first take a look at where the Airbnb accommodations in Paris are located.

We created a location distribution map that shows the number and percentage of listings in each of the 20 arrondissements of Paris. It turns out that **the northeast part of Paris is the hub of Airbnb accommodations**, with the **Buttes-Montmartre** (18th arrondissement) leading the pack with accommodations, accounting for 11.6% of the total listings. On the flip side, the **Louvre** (1st arrondissement) has the fewest accommodations, with only 802 listings, representing a mere 2.1% of the total accommodations.

To further explore the location distribution, we have included an interactive map below that provides more information. Simply hover your mouse over the map to uncover additional details!

<iframe src="/assets/images/2023-03-16-Paris-Airbnb-visualization/nlisting.html" width="100%" height="400px" frameborder="0"></iframe>

Are you curious to know the **exact locations** of these Airbnb accommodations in Paris? Try to click on the colorful circles in the interactive map, and voila! You can also zoom in and out for more detailed information, including the precise location, arrondissement, room type, price, and number of reviews for each unique accommodation!  
*(Quick heads up, it may take a moment for the map to load…*⌛️*)*

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9442dc33-0cf3-49d3-96a7-a7beaafc5f6f/listing_cluster.html" style="width:100%; height:500px; border:none;"></iframe>

# Which accommodations are the most popular, and during which time of the year?

To measure **popularity**, we combined the accommodation and review datasets, and calculated **the number of reviews** for each accommodation. The distribution of the number of reviews per accommodation is highly right-skewed, with the number of reviews ranging from 1 to 427. The 0.25 quantile is 2 reviews, the median is 6 reviews, and the 0.75 quantile is 33 reviews. To improve visualization, we removed the outliers with a z-score greater than 3.

![Distribution of number of reviews per accommodation](/assets/images/2023-03-16-Paris-Airbnb-visualization/nreview_dist.png)

Where are the popular accommodations located? We analyzed the average number of reviews for accommodations in each of the 20 arrondissements in Paris. We found that accommodations in **the center of Paris have the highest average number of reviews**, particularly in the **Louvre** (1st arrondissement) with average reviews reach 37.48, while **Menilmontant** (20th arrondissement) is at the lower end of the scale (18.85). Explore our interactive map to discover more!

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/169259fb-3a50-4ba9-8e60-36dcb962de61/review_map.html" style="width:100%; height:500px; border:none;"></iframe>

We also observed that the popularity of Airbnb accommodations in Paris increased annually. Airbnb started to get popular in Paris from 2014, and went to peak in June 2019 with over 31,000 reviews. However, the spread of COVID-19 in March 2020 led to a sharp decrease in reviews.

Additionally, the popularity of accommodations in Paris is highly **seasonal**, with the months of June, July, September, and October having the most reviews. Clearly, summer is the prime travel season!

Our interactive map with a **time slider** provides a more detailed analysis of how reviews change with time in different arrondissements!  
*(The time slider is on the top left corner, if it didn’t show up, please refresh the page)*

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3fbfa65-84f9-417a-a52f-58d4d06a41d1/review_time.html" style="width:100%; height:500px; border:none;"></iframe>

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b367a495-861e-4da8-b429-c362ab0644b1/review_slider.html" style="width:100%; height:500px; border:none;"></iframe>

# What are the prices **of Airbnb accommodations?**

The price is definitely something to consider when looking for an Airbnb accommodation! From cozy studios to luxurious apartments, Paris has it all. The prices of Airbnb accommodations in Paris vary greatly, with options ranging from 8€ to a jaw-dropping 12,000€! To get a better understanding of the price distribution, we analyzed the data and found that the 25th percentile of price is at 60€, the median is 88€, and the 75th percentile is at 125€.

Similar to the number of reviews, the price distribution is highly skewed to the right, so we removed outliers with a z-score greater than 3 for better visualization.

![Price distribution of Airbnb accommodations](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3f870321-6102-4297-b6c8-c8850e925c3a/price_dist.png)

But wait, there's more! We also analyzed the average prices of Airbnb accommodations in each of the 20 arrondissements in Paris. While accommodations in **the center of Paris are generally more expensive**, but the most expensive arrondissement is actually **Elysee** (8th arrondissement), with an average price of 204.08€. On the other hand, the cheapest arrondissement is **Menilmontant** (20th arrondissement), with an average price of 75.57€. The price difference between Elysee and Menilmontant is huge, with Elysee being 2.7 times more expensive than Menilmontant!

To get a better idea of the prices in each arrondissement, check out our interactive map below.

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b77f92f7-4739-4647-80e7-4c1f91fbcd7f/price_map.html" style="width:100%; height:500px; border:none;"></iframe>

> 💡 **A brief summary so far…**

- The majority of Airbnb listings in Paris are located in the northeast region, but tourists tend to book accommodations in the city center, even if they come at a higher cost.  
- Summer is the peak tourist season in Paris, although there are still travelers who visit during Christmas and new year.

---

# What amenities are provided in the accommodations?

When it comes to booking an Airbnb accommodation, it's important to make sure that you're getting all the amenities you need for a comfortable stay. From essential amenities like heating to conditional ones like a washing machine, it's important to check what the hosts are offering. And let's be real, who doesn't love finding some extra perks like hot tub and pool waiting for them?

To help you get a better sense of what amenities are commonly provided, we dug into the official Airbnb host page and found that amenities are divided into three categories: **essential**, **stand-out**, and **safety**. Check out the chart below to see the percentage of hosts providing each of these amenities. But here's the thing - we noticed that many hosts mention amenities that aren't listed on Airbnb's official page. So, we decided to compile a list of the most mentioned amenities and create a new category called "**high-demanding amenities**," which includes all those extra goodies that hosts frequently provide but aren't mentioned on the official list.

Take a closer look at the amenities that matter most to you and see how many hosts provide them!

![Essential amenities](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f1da8717-c8a0-4f93-9590-c23ae60615df/amenities_ess.png)

![Stand-out amenities](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/938b2f52-7fb7-4bc3-aada-327aeb005d36/amenities_so.png)

![Safety amenities](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7bb231da-c938-4ab7-8c0a-3876a99fc2a7/amenities_safe.png)

![High-demanding amenities](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82f10f3d-ad60-457f-ac26-f352de08f576/amenities_hd.png)

# **Short-term rentals and** Accommodation capacity

During our exploration of the Airbnb market in Paris, we noticed that some accommodations are not intended for short-term rentals. While most of them offer short-term rentals that **require a minimum stay** of less than a week, 5% of accommodations provide mid-term rentals that require a minimum booking of 8 to 30 nights. Moreover, only 1.15% of accommodations offer long-term rentals that require a minimum booking of at least 31 nights.

![Minimum nights for rentals](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/43b1a59e-8044-4bb5-b5b7-4425d80adeb1/minimum_nights.png)

In addition, we looked into the capacity of the accommodations and found that the majority of them only have one bedroom (74.33%), 84.86% of accommodations are suitable for groups of no more than four people.

![Bedroom and guest capacity](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3b73f488-f564-4bd6-bf07-83da63e529d6/bedroom_guest.png)

# Communication with hosts

When booking an Airbnb accommodation, a Superhost seems to be more reliable, and there are specific criteria to become one.

> [**How to become a Superhost**](https://www.airbnb.com/d/superhost)
> 
> - 4.8 or higher average overall rating based on reviews from their Airbnb guests in the past year.  
> - Completed at least 10 stays in the past year or 100 nights over at least 3 completed stays.  
> - Cancel less than 1% of the time.  
> - Respond to 90% of new messages within 24 hours.

We discovered that only 16.73% of Parisian hosts qualify as Superhosts, and the percentage varies by arrondissement. The **Louvre** (1st arrondissement) has the highest Superhost percentage at 26.31%, while the **Buttes-Chaumont** (19th arrondissement) has the lowest at 10.18%. **The popularity of each arrondissement seems to be closely related to its Superhost percentage**.

<iframe src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51e7d53e-fab0-4332-9cef-750ad5e7f2cc/superhost.html" style="width:100%; height:500px; border:none;"></iframe>

Since the response time of hosts counts as a key factor in qualifying as a Superhost, we examined the communication with Airbnb hosts. We found that **most hosts accept orders**, and 29.7% of accommodations are instantly bookable, meaning you can book them immediately without needing to send a request to the host for approval. Additionally, **hosts respond to most messages**, with over half (52.56%) of them responding within an hour.

![Acceptance rate](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe83ce62-63d5-434b-914a-9c6fefc68ddf/accept.png)

![Host response time](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/92430d75-eb07-4283-bef9-5a849fead9d3/host_resp.png)

---

Thank you for taking the time to read through our analysis of the Airbnb dataset in Paris. We hope that our findings have provided you with some insights into the Airbnb market in the city. We appreciate your interest in our work, and if you have any questions or feedback, please don't hesitate to contact us (📧 [Zixuan Tang](mailto:zixuantang.suki@gmail.com), 📧 [Ke Chen](mailto:kechen.kc94@gmail.com))!

Have a great day! 

> Python package used: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `geopandas`, `folium`
