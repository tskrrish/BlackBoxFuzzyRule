from sklearn.linear_model import LinearRegression

def fit_predictors(df):
    cols = df.columns.tolist()
    scores = {}
    for target in cols:
        inputs = [c for c in cols if c != target]      # the other three columns
        model = LinearRegression().fit(df[inputs], df[target])   # build a regression, train to predict target from other three coloumns
        scores[target] = model.score(df[inputs], df[target])     # to track R^2
    return scores

