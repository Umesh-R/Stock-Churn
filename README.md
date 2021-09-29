# Stock-Churn
This is a Django project that uses Monte Carlo Simulation to find profitable stock portfolio (with the stocks the user is interested in) with fixed funds. This mainly works with NSE stocks.

# Working in brief
The program fetches data from 1/1/2019 till date for the selected stocks, preprocesses it, calculates the exponentially weighted returns and variance in the prices of the stocks and then simulates over 2500 portfolios which are picked randomly (picks portfolios that are most likely to give profitable returns). The Volatility and Sharpe Ratio for each portfolio in each run is calculated and the Maximum Sharp Ratio portfolio (MSR) and the Global Minimum Volatility (GMV) portfolio are found. MSR, GMV along with other portfolios is shown in a graph, where the user can choose any portfolio and see its returns. 

# Django
The Django View and Model files are present in the folder Stock_Churn_Advisor and the Templates are in the template folder.

# Requirements
1. Django should be installed.
2. Numpy
3. Pandas
4. Matplotlib
5. nsepy

