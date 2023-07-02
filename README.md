# Retail_pricing

# Loan_repayment

## Content

* [Summary](README.md#Summary)  
* [Project description](README.md#Project-description)  
* [Data and methods](README.md#Data-and-methods)
* [ML model](README.md#ML-model)
* [Proposed algorythm](README.md#Proposed-algorythm)      
* [Project structure](README.md#Project-structure)                   


---

## Summary

It was proposed a new price optimization algorythm which is based on machine learning.  
  

## Project description

An online retailing company sales varous categories of goods. Its company's revenue as well as market share is decreasing over time because of emerging competitors which supply substitute goods with comparable quality. To tackle the problem, the management decided to take a set of actions. It has started with optimization of goods prices.

The business objective of the assignment is to identify factors, affecting price change, and optimize the pricing model.

## Data and methods

The dataset contains 676 unique purchases between 2017 and 2018 years. Each purchase has 30 features to study. 

The unit price is the target feature. It is essential, that it is discounted in case of wholesale as shown on the fig.1.

<div align="center"> <img src="./figures/fig_1.png" width="850">  </div>

 
The goods are spread by 52 gropups 


 



## ML model
 

## Proposed algorythm

 

## Project structure

<details>
  <summary>display project structure </summary>

```Python
Loan_repayment
├── .gitignore
├── config
│   └── config.json     # configuration settings
├── data                # data archive
│  
├── figures             # project figures and charts
│   ├── fig_1.png
.....
│   └── fig_xx.png
├── models              # models and weights
│   ├── xxx.pkl
.....
│   └── xxx.pkl
├── notebooks           # notebooks
│   └── Retail_pricing.ipynb

├── README.md
├── requirements.txt    
└── utils               # functions and data loaders
    └── reader_config.py
```
</details>
