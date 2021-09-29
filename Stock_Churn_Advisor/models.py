from django.db import models

# Create your models here.

class portfolio :
    weights : list
    fund_allocation : list
    names : list
    anual_returns : float
    anual_volatility : float
    sharpe_ratio : float
    length : list