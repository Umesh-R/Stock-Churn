from django.shortcuts import render
from django.http import HttpResponse
from numpy.lib.function_base import cov
from .models import portfolio as pf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
from datetime import date
from nsepy import get_history


# Create your views here.

simulations_df=None


def home(request):
    return render(request,'home.html')

def inputStocks(request):
    noOfStocks=int(request.POST['numberOfStocks'])
    funds=int(request.POST['funds'])
    minInvestment=int(request.POST['minInvestment'])
    request.session['Number_of_stocks']=noOfStocks
    request.session['Funds']=funds
    request.session['Min_investment']=minInvestment
    noOfStocksArr=[i for i in range(1,noOfStocks+1)]
    return render(request,'inputStockNames.html',{'numOfStocks':noOfStocks,'arr':noOfStocksArr})

def processStocks(request):
    '''TEST'''
    #simulations_df = pd.read_pickle('E:\\Projects\\Stock_Churn\\static\\simulations_df.pkl')
    #names=['ZEEL.csv','WIPRO.csv','POWERGRID.csv','SBIN.csv','TCS.csv','JSWSTEEL.csv']


    #GET DATA
    number_of_stocks,names,fund,min_weight = get_input(request)
    perc_change,latest_price = get_data(names)
    stocks_mean,stocks_cov = get_mean_cov(perc_change,freq=252,cspan=252,mspan=500)

    #simualtion
    simulations_df = simulation(number_of_stocks,stocks_mean,stocks_cov,fund,min_weight,latest_price) 

    # Return the Max Sharpe Ratio from the run.
    max_sharpe_ratio = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]
    # Return the Min Volatility from the run.
    min_volatility = simulations_df.loc[simulations_df['Portfolio Risk'].idxmin()]

    print('')
    print('='*80)
    print('MAX SHARPE RATIO:')
    print('-'*80)
    MSR_portfolio = print_portfolio(max_sharpe_ratio,names)
    print('-'*80)

    print('')
    print('='*80)
    print('MIN VOLATILITY:')
    print('-'*80)
    GMV_portfolio = print_portfolio(min_volatility,names)
    print('-'*80)    


    sims=simulations_df.copy()
    sims.columns=["y","x","Sharpe Ratio","Weights","Fund Distribution"]

    columns_titles = ["x","y","Sharpe Ratio","Weights","Fund Distribution"]
    
    sims=sims.reindex(columns=columns_titles)
    data = sims.to_json(orient='records')

    var=[]
    for i in names:
        var.append(stocks_cov[i][i]*100)
    
    return render(request,'output.html',{'MSR': MSR_portfolio, 'GMV': GMV_portfolio ,'data':data , 'stocks':names, 'mean':stocks_mean*100, 'cov':var})  

def get_input(request):
    number_of_stocks=request.session['Number_of_stocks']
    fund=request.session['Funds']
    min_weight=request.session['Min_investment']/100
    names=[]
    for i in range(1,number_of_stocks+1):
        names.append(request.POST['stock_'+ str(i) ])
    
    return number_of_stocks,names,fund,min_weight



def get_data(names):
    perc_change=pd.DataFrame()
    data = pd.DataFrame()
    for stock in names:
        df =  get_history(symbol=stock,start=date(2019,1,1),end=date.today())
        df=df['Close']
        #data is a DF of stock's data
        data[stock]=df
        #perc_change is a the DF of different stock's perc change in one month data
        perc_change=pd.concat([perc_change,df.pct_change()],axis=1)   
    perc_change.columns=names
    latest_price=data.iloc[-1]
    return perc_change,latest_price

def get_fund_dist(weights,fund):
    return np.array(fund*weights).astype(int)

def print_portfolio(portfolio,names):
    ind,=np.where(portfolio['Weights']>0)
    stocks_used,funds_distribution,weights=[],[],[]
    print('PORTFOLIO WEIGHTS')
    for i in ind:
            stocks_used.append(names[i])
            funds_distribution.append(portfolio['Fund Distribution'][i])
            weights.append(portfolio['Weights'][i]*100)
            print(names[i],'\t:',portfolio['Weights'][i]*100,'%')
    print('')        
    print('FUND ALLOCATION')
    for i in ind:
            print(names[i],'\t:',portfolio['Fund Distribution'][i])        
    print('')
    print('PORTFOLIO ANUAL RETUNS (EXPECTED) : ',portfolio['Portfolio Return']*100,'%')
    print('')
    print('PORTFOLIO VOLITALITY (EXPECTED): ',portfolio['Portfolio Risk']*100,'%')
    print('')
    print('SHARPE RATIO : ',portfolio['Sharpe Ratio']) 

    funds_distribution=zip(stocks_used,funds_distribution)
    weights=zip(stocks_used,weights)

    portfolioX=pf()
    portfolioX.anual_returns = portfolio['Portfolio Return']*100
    portfolioX.anual_volatility = portfolio['Portfolio Risk']*100
    portfolioX.fund_allocation =  funds_distribution
    portfolioX.names = stocks_used
    portfolioX.weights = weights
    portfolioX.sharpe_ratio =  portfolio['Sharpe Ratio']   
    portfolioX.length= [x for x in range(len(stocks_used))]     

    return portfolioX
    
def get_weights(rnd,number_of_stocks,stocks_mean,min_weight,latest_price,fund):
    
    w = np.array(rnd.random(number_of_stocks))
    w*=stocks_mean
    w[w<0]=0
    w/=w.sum()
    #redistribution of lesser weights
    greater = (w>=min_weight)
    lesser = (w<min_weight)
    
    #change weights if all weights are lesser than min weight
    #possibility of infinite loop
    if(len(greater)==0):
        get_weights(rnd,number_of_stocks,stocks_mean,min_weight,latest_price,fund)
    
    #check if there are weights below min weight
    #check if the weight of min weight should be left alone, added evenly to the rest or added based on the mean
    #try except might be needed possibility of division by 0
    w[greater]+=(w[lesser].sum())/len(w[greater])
    w[lesser]=0
    w/=w.sum()

    #change weights if a single stock cannot be bought with the allocation
    fund_dist=get_fund_dist(w,fund)
    vol=fund_dist/latest_price
    #possibility of infinite loop
    if(not (np.array_equal(w>0,vol>=1))):
        get_weights(rnd,number_of_stocks,stocks_mean,min_weight,latest_price,fund)
    
    return w,fund_dist    



def  get_exp_cov(X,Y,span):
    covariation= (X-X.mean())*(Y-Y.mean())
    return covariation.ewm(span=span).mean().iloc[-1]

def get_mean_cov(perc_change,freq,cspan,mspan):
    #For exp weighting
    mean= perc_change.ewm(span=mspan).mean().iloc[-1]*freq
    #find covariatoin between stocks and then ewm to find exp weighted covariance
    number_of_stocks=len(perc_change.columns)
    names = perc_change.columns
    temp = np.zeros([number_of_stocks,number_of_stocks])
    for i in range(number_of_stocks):
        for j in range(i,number_of_stocks):
            temp[i,j] = temp[j,i] = get_exp_cov(perc_change.iloc[:,i],perc_change.iloc[:,j],cspan)
    
    cov=pd.DataFrame(temp*freq,columns=names,index=names) 

    return mean,cov


def simulation(number_of_stocks,stocks_mean,stocks_cov,fund,min_weight,latest_price):
    print('number of stocks: ',number_of_stocks,'\nMean: ',stocks_mean,'\nCOV: ',stocks_cov,'\nFund: ',fund,'\n Min weight: ',min_weight,'\nlatest prices: ',latest_price)
    rnd=np.random.default_rng()
    # simulation will run with 5000 iterations.
    num_of_portfolios = 2500
    # all_weight is an array to store the weights as they are generated.
    all_weights = np.zeros((num_of_portfolios, number_of_stocks))
    # all_fund_dist is an array to store the funds distribution as they are generated.
    all_fund_dist = np.zeros((num_of_portfolios, number_of_stocks))
    # Prep an array to store the returns as they are generated.
    ret_arr = np.zeros(num_of_portfolios)
    # Prep an array to store the volatilities as they are generated.
    vol_arr = np.zeros(num_of_portfolios)
    # Prep an array to store the sharpe ratios as they are generated.
    sharpe_arr = np.zeros(num_of_portfolios)
    #simulation
    for i in range(num_of_portfolios):
        #get weights
        weights,fund_dist= get_weights(rnd,number_of_stocks,stocks_mean,min_weight,latest_price,fund)
        #adding weights to all weights array
        all_weights[i,:]=weights
        #adding fund_dist to all_fund_dist
        all_fund_dist[i,:]=fund_dist
        #calcualte the retuns 
        ret_arr[i]= np.sum(stocks_mean*weights)
        #calculate the volitality
        vol_arr[i]=np.sqrt(np.dot(weights.T,np.dot(stocks_cov,weights)))
        #calculating the sharpe ration
        sharpe_arr[i] = ret_arr[i]/vol_arr[i]
        
    #creating the simulation DF
    simulations_df = pd.DataFrame(data=[ret_arr, vol_arr, sharpe_arr, all_weights,all_fund_dist]).T
    simulations_df.columns=['Portfolio Return','Portfolio Risk','Sharpe Ratio','Weights','Fund Distribution']   
    # Make sure the data types are correct, we don't want our floats to be strings.
    simulations_df = simulations_df.infer_objects()
    return simulations_df
