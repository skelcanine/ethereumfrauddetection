import pandas as pd


def uekaggleEuropeanCCFraud():

    file_path = "kaggleEuropeanCCFraud.csv"
    return pd.read_csv(file_path)

def euekaggleEuropeanCCFraud():
    file_path = "kaggleEuropeanCCFraud.csv"
    # Import dataset
    df_main = pd.read_csv(file_path)
    # Drop Time variable
    df = df_main.drop(['Time'], axis=1)
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Split data as class and other parameters
    X = df.drop('Class', axis=1)
    y = df['Class']
    return [X, y]

def uekaggleEthereumFraud():
    file_path = "kaggleEthereumFraud.csv"
    return pd.read_csv(file_path)

def ekaggleEthereumFraud():
    file_path = "kaggleEthereumFraud.csv"
    df = pd.read_csv(file_path)
    # print(df.shape)

    # Remove columns with only 1 value
    uniqueness = 1
    dfe = df.loc[:, (df.nunique() > uniqueness)]

    #Clean rows with N/A values
    # print(dfe.shape)
    dfe = dfe.dropna()
    dfe = dfe.reset_index(drop=True)
    # print(dfe.shape)



    # Drop unnecessary columns
    dfe = dfe.drop(
        columns=['Unnamed: 0', 'Index', 'Address', ' ERC20_most_rec_token_type', ' ERC20 most sent token type'])
    """
    Correlation
    
    plt.rcParams["figure.figsize"] = (50, 50)
    corr_matrix = dfe.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()
    """

    # print(dfe.columns)
    """
    ANOVA
    
    for col in dfe.columns:
        if col != 'FLAG':
            print('@@@@@@@@@@'+col)
            multiComp = MultiComparison(dfe[col], dfe['FLAG'])
            tukeyres = multiComp.tukeyhsd(alpha=0.05)

            print(tukeyres)
    """
    # Separate fraud flag

    dfe = dfe.drop_duplicates()

    y = dfe['FLAG']
    dfe = dfe.drop(columns=['FLAG'])

    return [dfe,y]



