# python xgboost explainer

### inspired by [xgboostExplainer of R language](https://github.com/AppliedDataSciencePartners/xgboostExplainer). addtionally fixed [a bug about lambda](https://github.com/AppliedDataSciencePartners/xgboostExplainer/pull/5)

### usage:
```python
...
# calculate logit-odds of each node of each tree
tree_lst = xgb_exp.model2table(bst)

...

leaf_lst = bst.predict(sample, pred_leaf=True)
# sum the logit-odds contribution of each feature
dist = xgb_exp.logit_contribution(tree_lst, leaf_lst[0])

# print result
sum_logit = 0.0
for k in dist:
    sum_logit += dist[k]
    fn = feature_map[int(k[1:])] if k != "intercept" else k
    print(fn + ":", dist[k])
```

### sample result
> intercept: -1.4477233142227064
> 
> last_evaluation: -0.7779663104556692
> 
> average_montly_hours: -0.4159895659998113
> 
> time_spend_company: -0.25305760508286557
> 
> satisfaction_level: -0.6634059629069854
> 
> number_project: 0.02692557805173946
> 
> Work_accident: -1.2246923984381386
> 
> salary: 0.17029401462985444
> 
> sales: -0.023232836934958548
> 
> promotion_last_5years: 0.009318957559541447

### you can check [demo notebook](https://github.com/gameofdimension/xgboost_explainer/blob/master/xgboost_explainer_demo.ipynb) for a full example 


