# Author: Nana Owusu
# Purpose: To store useful functions that aid in my
# machine learning experience.

from numpy import (sqrt, tanh, arctanh, transpose,
    ndarray, array, linspace, round, zeros, ones)
from sklearn.metrics import (roc_curve, roc_auc_score,
    confusion_matrix)
from sklearn.linear_model import LogisticRegressionCV
from scipy.optimize import curve_fit
from scipy.stats import norm, spearmanr, pearsonr, f
from matplotlib.pyplot import plot, legend
from pandas import ExcelWriter, DataFrame, Series
from matplotlib.figure import Figure
# from functools import reduce
# from itertools import repeat, starmap

def LOO_testing(data:DataFrame, target:str, model:str, predictor:Series, 
                response:Series, classifier:LogisticRegressionCV):
    """ Leave-One-Out cross-validation method for
    testing a regression model. There is an outer
    and inner method for leaving one out, thus
    there are really two left out.
    
    Input
    -----
    data:DataFrame: a Pandas object containing relevant 
                    model data
    predictor:Series: a Pandas object of single column data
                    to be used as a predictor for the model
    response:Series: a Pandas object of single column data
                    to be used as a response being modeled
    
    Output
    ------
    Produces a generator (which is iterable) containing the
    probability of the model matching the test datum which
    was left out.
    """
    
    for k in data.index:
        #Leaving One Out For Testing
        X_train = predictor.drop(k,axis=0)
        X_test = predictor.loc[k]
        Y_train = response.drop(k,axis=0)
        Y_test = response.loc[k]
        pos = Y_train.sum()

        #LOOCV, Training and Validation
        classifier.fit(X_train.values,Y_train.values)
        proba = classifier.predict_proba(X_test.values.reshape((1, -1)))[0,1]
                        
        #Storing Prediction On Test Set
        data.loc[k,target+'_'+model+'_proba'] = proba
        data.loc[k,target+'_'+model+'_pred'] = int(proba>=0.5)
    
    #Formatting Predictions
    data[f'{target}_{model}_pred']=\
    data[f'{target}_{model}_pred'].astype('int64')

def LOOwCO_testing(data:DataFrame, target:str, model:str, predictor:Series, 
                   response:Series, cutoffs:ndarray, classifier:LogisticRegressionCV):
    """ Leave-One-Out cross-validation method for
    testing a regression model. There is an outer
    and inner method for leaving one out, thus
    there are really two left out. A cutoff value
    is determined 
    
    Input
    -----
    data:DataFrame: a Pandas object containing relevant 
                    model data
    predictor:Series: a Pandas object of single column data
                    to be used as a predictor for the model
    response:Series: a Pandas object of single column data
                    to be used as a response being modeled
    
    Output
    ------
    Produces a generator (which is iterable) containing the
    probability of the model matching the test datum which
    was left out.
    """
    
    for k in range(len(data)):
        #Leaving One Out For Testing
        X_train = predictor.drop(k,axis=0)
        X_test = predictor.loc[k]
        Y_train = response.drop(k,axis=0)
        
        pos = Y_train.sum()

        #LOOCV, Training and Validation
        classifier.fit(X_train.values,Y_train.values)
        proba = classifier.predict_proba(X_train).values[:,1]
        cutoff = fnr_cutoff(cutoffs, proba, Y_train, pos, 0.05)[0]
                        
        #Storing Prediction On Test Set
        proba = classifier.predict_proba(X_test.values.reshape((1, -1)))[0,1]
        data.loc[k,target+'_'+model+'_proba'] = proba
        data.loc[k,target+'_'+model+'_pred'] = int(proba>=cutoff)
    
    #Formatting Predictions
    data[f'{target}_{model}_pred']=\
    data[f'{target}_{model}_pred'].astype('int64')

def fnr_cutoff(cutoffs:ndarray, y_prob:Series|ndarray, y_train:Series|ndarray, pos:int, fnr_min:float):
    for cutoff in cutoffs:
        #Getting Predicted Labels
        y_pred=(y_prob>=cutoff).astype('int64')

        #Computing FNR, break at 5% With cutoff
        tru_neg, fls_pos, _, _ = \
        confusion_matrix(y_train,y_pred,normalize=None).ravel()
        if (fls_pos/pos)>fnr_min:
            continue
        else:
            return (cutoff, tru_neg)

###############################
#### Statistical functions ####
###############################

def add_xtoy(X:Series, y:Series)->float:
    """ Subroutine used for calculating the DeLong test
    statistic. Copied from Kyle's library and modified by
    me.
    """
    
    m = len(X)
    result = 0
    for val in X:
        result += rank_mw(val, y)
        
    return result / m

def add_ytox(x:float, Y:Series)->float:
    """ Subroutine used for calculating the DeLong test
    statistic. Copied from Kyle's library and modified by
    me.
    """
    
    n = len(Y)
    result = 0
    for val in Y:
        result += rank_mw(x, val)
        
    return result / n

def rank_mw(a:float, b:float)->int|float:
    """ Subroutine used for calculating the DeLong test
    statistic. Copied from Kyle Kalutkiewicz's library and
    modified by me.
    
    Related to the Mann-Whitney U-statistic.
    """
    
    try:
        a < b
    except TypeError:
        print("Both inputs must be numbers.\n")
        raise TypeError
    else:
        if a < b:
            return 0
        elif a > b:
            return 1
        elif a == b:
            return 0.5
        
def auc_mw(x:ndarray, y:ndarray)->list:
    """ subroutine used for calculating the delong test
    statistic. Copied from Kyle's library and modified
    by me.
    
    This function calculates the averages of values
    from the Mann-Whitney U-statistic equivalent of
    """
    
    m, n = len(x), len(y)
    rank_sum = 0
    for x_val in x:
        for y_val in y:
            rank_sum += rank_mw(x_val, y_val)

    
    # for val in x:
    #     rank_sum += reduce(lambda val1, val2: val1 + val2, \
    #                        starmap(rank_mw, zip(repeat(val,n), y)))/(m*n)
            
    return rank_sum / (m*n)
                               
def sums_10(x1:list|ndarray, y1:list|ndarray, x2:list|ndarray, y2:list|ndarray)->list:
    """ Subroutine used for calculating the delong test
    statistic. Copied from Kyle's library and modified
    by me.
    """
    
    AUC1 = auc_mw(x1, y1)
    AUC2 = auc_mw(x2, y2)
    m = len(x1)
    
    S10_11=sum([(add_ytox(x1[i],y1)-AUC1)*(add_ytox(x1[i],y1)-AUC1) for i in range(m)])
    S10_12=sum([(add_ytox(x1[i],y1)-AUC1)*(add_ytox(x2[i],y2)-AUC2) for i in range(m)])
    S10_21=sum([(add_ytox(x2[i],y2)-AUC2)*(add_ytox(x1[i],y1)-AUC1) for i in range(m)])
    S10_22=sum([(add_ytox(x2[i],y2)-AUC2)*(add_ytox(x2[i],y2)-AUC2) for i in range(m)])

    return (1/(m-1))*array([[S10_11,S10_12],[S10_21,S10_22]])

def sums_01(x1:list|ndarray, y1:list|ndarray, x2:list|ndarray, y2:list|ndarray)->list:
    """ Subroutine used for calculating the delong test
    statistic. Copied from Kyle's library and modified
    by me.
    """
    
    AUC1 = auc_mw(x1, y1)
    AUC2 = auc_mw(x2, y2)
    n = len(y1)
    
    S01_11=sum([(add_xtoy(x1,y1[i])-AUC1)*(add_xtoy(x1,y1[i])-AUC1) for i in range(n)])
    S01_12=sum([(add_xtoy(x1,y1[i])-AUC1)*(add_xtoy(x2,y2[i])-AUC2) for i in range(n)])
    S01_21=sum([(add_xtoy(x2,y2[i])-AUC2)*(add_xtoy(x1,y1[i])-AUC1) for i in range(n)])
    S01_22=sum([(add_xtoy(x2,y2[i])-AUC2)*(add_xtoy(x2,y2[i])-AUC2) for i in range(n)])

    return (1/(n-1))*array([[S01_11,S01_12],[S01_21,S01_22]])

def auc_ci(auc:float, pos:int, neg:int, alpha:float)->tuple:
    """Returns the lower and upper confidence intervals for estimating an AUC
    Calculated assuming AUC is normally distributed (approximate for large datasets)
    Copied from Kyle Kalutkiewicz's library.
    """

    q_pos = auc/(2-auc)
    q_neg = 2*(auc**2)/(1+auc)
    std_err = sqrt((auc*(1-auc)+(pos-1)*(q_pos-auc**2)+(neg-1)*(q_neg-auc**2))/(pos*neg))
    
    return (auc-std_err*norm.ppf(1-0.5*alpha),auc+std_err*norm.ppf(1-0.5*alpha))

def calc_auc(data:DataFrame, y_true:Series, y_pred:Series, alpha:float)->tuple:
    """ Calculate the area-under-the-curve (AUC) which
    quantifies the performance of the predictive model.
    Include the lower and upper bounds of the confidence
    interval for the AUC.
    
    Input
    -----
    data:DataFrame: a Pandas object containing relevant 
                    model data
    y_true:Series: a Pandas object of the response being
                    modeled.
    y_pred:Series: a Pandas object of the predicted
                    response values.
    """
    
    positives = y_true.sum()
    negatives = len(data) - positives
    auc = roc_auc_score(y_true, y_pred)
    auc_L, auc_U = auc_ci(auc,positives,negatives,alpha)
    
    return (auc_L, auc, auc_U)
    
def DeLong2_test(x1:ndarray, y1:ndarray, x2:ndarray, y2:ndarray)->tuple:
    """ Perform DeLong test which evaluates two
    different area under the curve (AUC) values.
    """
    
    L = transpose(array([[1, -1]]))
    m, n = len(x1), len(y1)
    AUC1 = auc_mw(x1, y1)
    AUC2 = auc_mw(x2, y2)
    root = sqrt(transpose(L).dot(((1/m)*sums_10(x1,y1,x2,y2) + \
                                  (1/n)*sums_01(x1,y1,x2,y2)).dot(L)))
    z_score = ((AUC1 - AUC2)/root)[0][0]
    pvalue = 1 - norm.cdf(abs(z_score))
    
    return (z_score, pvalue)

def binomial_ci(x:ndarray, n:int, alpha:float)->tuple:
    """ Copied from Kyle's library.
    """
    
    F_L = f.ppf(0.5*alpha, 2*x, 2*n - 2*x + 2)
    F_U = f.ppf(1-0.5*alpha, 2*x + 2, 2*n - 2*x)
    
    if x==0:
        p_L = 0
    else:
        p_L = 1/(1 + ((n - x + 1)/(x*F_L)))
    
    if x==n:
        p_U = 1
    else:
        p_U = 1/(1 + ((n - x)/((F_U*x + F_U))))
    
    return p_L, p_U

def pearsonr_ci(x:ndarray, y:ndarray, alpha:float)->tuple:
    """Calculate Pearson r correlation as well as 
    its confidence interval.
    
    Input
    -----
    x:ndarray: vector of abscissa values
    y:ndarray: vector of range values
    alpha:float: false positive rate
    
    Output
    ------
    r:float: Pearson r quantity
    r_low:float: low value of confidence interval at set alpha
    r_high:float: high value of confidence interval at set alpha
    """
    
    # compute the Pearson r correlation
    r, p = pearsonr(x,y)
    
    # convert r quantity to z statistic
    r_z = arctanh(r)
    
    # compute standard error
    std_err = 1/sqrt(x.size-3)
    
    # calculate z statistic at desired alpha
    z = norm.ppf(1-alpha/2)
    
    # calculate the low and high extexts of the 
    # confidence interval
    low_z, high_z = r_z-z*std_err, r_z+z*std_err
    r_low, r_high = tanh((low_z, high_z))
    
    return r, r_low, r_high

def spearmanr_ci(x:ndarray, y:ndarray, alpha:float)->tuple:
    """Calculate Spearman rho correlation as well as 
    its confidence interval.
    
    Attributes
    ----------
    x:ndarray: abscissa of the data
    y:ndarray: range of the data
    alpha:float: false positive rate
    
    Return
    ------
    rho:float: Pearson r quantity
    rho_low:float: low value of confidence interval at set alpha
    rho_high:float: high value of confidence interval at set alpha
    """
    
    # compute the Pearson r correlation
    rho = spearmanr(x,y)[0]
    
    # calculate the low and high extexts of the 
    # confidence interval
    rho_low = tanh(arctanh(rho) - sqrt((1+0.5*(rho**2))/(len(x)-3))
                     * norm.ppf(1-(alpha/2)))
    rho_high = tanh(arctanh(rho) + sqrt((1+0.5*(rho**2))/(len(x)-3))
                       * norm.ppf(1-(alpha/2)))
    
    return rho, rho_low, rho_high

def classification_metrics(data:DataFrame, y_true:Series, y_pred:Series, alpha:int|float)->dict:
    """ Function for populating the metrics of classification
    algorithms.
    """
    
    # Calculate the confusion matrix from
    # the predicted and true values. 
    tru_neg, fls_pos, fls_neg, tru_pos = \
    confusion_matrix(y_true, y_pred, normalize=None).ravel()
    
    # Derive the negative and positive
    # predictive values (NPV/PPV) as well
    # as the sensitivity and specificity.
    pos = y_true.sum()
    neg = len(data) - pos
    NPV = tru_neg/(tru_neg+fls_neg)
    PPV = tru_pos/(tru_pos+fls_pos)
    sens = tru_pos/pos
    spec = tru_neg/neg
    
    # Calculate the upper and lower
    # bounds of the sensitivity, specificity,
    # NPV, and PPV.
    sens_L, sens_U = binomial_ci(tru_pos,pos,alpha)
    spec_L, spec_U = binomial_ci(tru_neg,neg,alpha)
    NPV_L, NPV_U = binomial_ci(tru_neg,tru_neg+fls_neg,alpha)
    PPV_L, PPV_U = binomial_ci(tru_pos,tru_pos+fls_pos,alpha)
    
    class_dict = {'TP':tru_pos, 'TN':tru_neg, 'FP':fls_pos,
                  'FN':fls_neg, 'sens_L':sens_L, 'sens':sens, 
                  'sens_U':sens_U, 'spec_L':spec_L, 'spec':spec,
                  'spec_U':spec_U, 'NPV_L':NPV_L, 'NPV':NPV, 
                  'NPV_U':NPV_U, 'PPV_L':PPV_L, 'PPV':PPV, 'PPV_U':PPV_U}
    
    return  class_dict

def youden_cutoff(y_true:Series, y_pred:Series)->DataFrame:
    """ Produce the confusion matrix of the
    true and predicted outcomes. With that, calculate
    the sensitivity, specificity, Youden index, and
    balanced accuracy from the two outcomes.
    
    Input
    -----
    y_true:Series: true outcomes
    y_pred:Series: predicted outcomes
    
    Output
    ------
    sens:float64: sensitivity
    spec:float64: specificity
    youden_idx:float64: Youden Index
    bal_acc:float64: balanced accuracy
    """
    
    # Calculate the confusion matrix from
    # the predicted and true values. 
    tru_neg, _, _, tru_pos = \
    confusion_matrix(y_true, y_pred, normalize=None).ravel()
    
    # Derive the negative and positive
    # as well as the sensitivity and specificity.
    pos = y_true.sum()
    neg = len(y_true) - pos
    sens = tru_pos/pos
    spec = tru_neg/neg
    youden_idx = sens + spec - 1
    
    # balanced accuracy
    bal_acc = 0.5 * (youden_idx + 1)
    
    return (sens, spec, youden_idx, bal_acc)

def balanced_accuracy(y_true:Series, mre_data:DataFrame|ndarray, cutoffs:DataFrame|ndarray)->DataFrame:
    """ Determine the balanced accuracy of an MRE 
    parameter that is given.
    
    Attributes
    ----------
    y_true:Series: true outcomes
    mre_data:ndarray: desired MRE parameter
    cutoffs:ndarray: parameter range at which to assess accuracy
    
    Return
    ------
    ba_output:DataFrame: sensitivity, specificity, Youden Index,
                        and balanced accuracy for each cutoff
    """
    
    # Storage for calculated accuracy values
    ba_output = DataFrame(index=cutoffs, 
                          columns=['sens','spec','Youden Index',
                                   'Balanced Accuracy'])
    for cutoff in cutoffs:

        y_pred = Series(mre_data>=cutoff).astype('int16')
        sens, spec, youden_idx, ba = \
        youden_cutoff(y_true, y_pred)

        ba_output.loc[cutoff,'sens'] = sens
        ba_output.loc[cutoff,'spec'] = spec
        ba_output.loc[cutoff,'Youden Index'] = youden_idx
        ba_output.loc[cutoff,'Balanced Accuracy'] = ba
        
    return ba_output

def net_benefit(true_pred:DataFrame|ndarray, probabilities:list|tuple, start_threshold:int|float=0.01, 
                end_threshold:int|float=0.99, pt_len:int=100, nbtype:str='treated')->DataFrame:
    """Calculate the net benefit of probabilities generated by a predictive model; to be used for
    decision curve analysis (DCA). Options for 'treated', 'untreated', 'overall', and 'adapt'
    benefit analyses are supported.
    
    Parameters
    ----------
    true_pred:DataFrame|ndarray
        Categorical data to which the probabilities are related. 
    probabilities:list|tuple
        Iterable datatype of probabilites on which to perform DCA
    start_threshold:int|float
        Value at which to start the probability threshold
    end_threshold:list|tuple
        Value at which to end the probability threshold
    pt_len:int
        Number representing how many thresholds to test between star and end values
    nbtype:str
        Options for net benefit analysis: treated, untreated, overall, adapt
    
    Return
    ------
    netbenefit:DataFrame
        Object containing thresholds, and net benefits for treating all, none, and
        according to the model probabilities given.
    """

    pop_size = len(true_pred)
    netbenefit = DataFrame(columns=['threshold','all','none'])
    proba_threshold = linspace(start_threshold, end_threshold, pt_len)

    coeff = zeros((2, pt_len))
    net_benefit_treated = zeros(pt_len)
    net_benefit_untreated = zeros(pt_len)

    match nbtype:
        case 'treated':
            coeff[0,:] = array([1] * pt_len)
        case 'untreated':
            coeff[1,:] = array([1] * pt_len)
        case 'overall':
            coeff = ones((2, pt_len))
        case 'adapt':
            coeff[0,:] = 1-proba_threshold
            coeff[1,:] = proba_threshold
        case _:
            print(f'The decistion curve analysis nbtype must be: treated, untreated,'
                  +'overall, or adapt. {nbtype} is not an option')
            raise RuntimeError

    event_rate = true_pred.sum() / pop_size
    net_benefit_all = (event_rate) - (1-event_rate)*(proba_threshold/(1-proba_threshold))
    net_benefit_none = (1-event_rate) - (event_rate)*((1-proba_threshold)/proba_threshold)

    for count, item in enumerate(probabilities):
        for i, threshold in enumerate(proba_threshold):
            tp = sum((probabilities[item] >= threshold) & (true_pred == 1))
            fp = sum((probabilities[item] >= threshold) & (true_pred == 0))
            fn = sum((probabilities[item] < threshold) & (true_pred == 1))
            tn = sum((probabilities[item] < threshold) & (true_pred == 0))

            net_benefit_treated[i] = tp/pop_size - (fp/pop_size) * (threshold/(1-threshold))
            net_benefit_untreated[i] = tn/pop_size - (fn/pop_size) * ((1-threshold)/threshold)

        netbenefit[f'prediction{count}'] = \
        Series(coeff[0,:]*net_benefit_treated + coeff[1,:]*net_benefit_untreated)

    netbenefit['threshold'] = Series(proba_threshold)
    netbenefit['all'] = Series(coeff[0,:]*net_benefit_all)
    netbenefit['none'] = Series(coeff[1,:]*net_benefit_none)
    
    return netbenefit

##########################
### Plotting functions ###
##########################

def roc_plot(fig:Figure, data:DataFrame, target:str, model_list:list|tuple, 
             models:list, title:str):
    for model in model_list:
        y_true = data[target]
        y_prob = data[target + '_' + model + '_proba']
        # False-Positive rate (FPR),
        # True-Positive rate (TPR), and threshold of
        # the ROC curve. thresholds
        FPR,TPR,_ = roc_curve(y_true,y_prob)
        
        # Area under the Receiver-Operating
        # Characteristic (ROC) curve.
        AUC = roc_auc_score(y_true,y_prob)
        
        # Plotting
        ax_range=[0.0,1.0]
        roc_range = linspace(0.0, 1.0, 10)
        tenths=[n*0.1 for n in range(11)]
        
        fig.axes.append(
            plot(FPR,TPR,label='+'.join(models[model])+', AUC='+str(round(AUC,4)),
                linewidth=4))
        
    fig.axes.append(
        plot(roc_range, roc_range, color='k', linestyle='dotted'))
    fig.axes[0].set_xlabel('1-Specificity', fontsize=16)
    fig.axes[0].set_ylabel('Sensitivity', fontsize=16)
    fig.axes[0].grid(visible=True, which='both')
    fig.axes[0].set_xticks(tenths)
    fig.axes[0].set_yticks(tenths)
    fig.axes[0].set_xlim(ax_range)
    fig.axes[0].set_ylim(ax_range)
        
    legend(loc='lower right', fontsize=16)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

def dca_plot(fig:Figure, net_benefits:DataFrame, title:str):
         
    fig.axes.append(
        plot(net_benefits['threshold'], net_benefits['all'], color='r', 
                 linestyle='solid', label='all'))
    fig.axes.append(
        plot(net_benefits['threshold'], net_benefits['none'], color='g', 
                 linestyle='dotted', label='none'))
    fig.axes.append(
        plot(net_benefits['threshold'], net_benefits['prediction0'], color='#009d94', 
                 linestyle='dashdot', label='pred 1'))
    fig.axes.append(
        plot(net_benefits['threshold'], net_benefits['prediction1'], color='purple', 
                 linestyle='dashed', label='pred 2'))
    
    fig.axes[0].set_ylim((-0.1, net_benefits[['all', 'prediction0', 'prediction1']].max().max()))
    fig.axes[0].set_xlabel('Threshold Probability', fontsize=16)
    fig.axes[0].set_ylabel('Net Benefit', fontsize=16)
    fig.axes[0].grid(visible=False, which='both')
    
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

###############################
### Save Dataframe as Excel ###
###############################

def export_as_excel(outname:str, a_or_w:str, list_of_df:list|tuple, sheet_name:dict, use_idx:bool=False):
    """ Given the compiled Pandas DataFrame objects,
    this function saves the objects as Excel workseets
    in a single spreadsheet.
    
    Input
    -----
    outname:str: Name of the file with or without the address
                to be saved.
    a_or_w:str: append 'a' or write 'w'
    list_of_df:list: list of DataFrames objects
    sheet_dict:dict: Key, value pair containing the name and
                    the associated Pandas object.
    use_idx:bool: include DataFrame indices or not
    """
    
    with ExcelWriter(outname, engine='openpyxl', mode=a_or_w) as writer:
        for sheet in list_of_df:
            sheet.to_excel(writer, sheet_name=sheet_name, index=use_idx)