import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preproces_data(URL, test_size_fraction):
    """
    Load and preprocess the dataset from the given CSV URL, selecting relevant features for training a regression model to predict property prices.

    - Reads the dataset from the specified CSV file.
    - Selects a predefined subset of features (Group 1) considered important for the model.
    - Separates the target variables ('PRICE' and 'UNITPRICE') from the predictors.
    - Handles class stratification by splitting the data into frequent and rare target price classes to ensure a balanced train-test split.
    - Returns the training and testing sets for both features and target.

    :param URL: Path or URL to the CSV dataset file.
    :type URL: str
    :param test_size_fraction: Fraction of the dataset to include in the test split.
    :type test_size_fraction: float
    :return: Four pandas DataFrames: X_train, X_test, y_train, y_test
    :rtype: tuple
    """

    # Load the data
    df = pd.read_csv(URL)

    # Select important feautres
    # Group 2
    '''df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
                 'HASTERRACE', 'HASLIFT', 'ISPARKINGSPACEINCLUDEDINPRICE',
                 'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL', 'HASGARDEN',
                 'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN',
                 'PERIOD_201803', 'PERIOD_201806', 'PERIOD_201809', 'PERIOD_201812']]'''

    # Group 1
    df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
                 'HASTERRACE', 'ISPARKINGSPACEINCLUDEDINPRICE',
                 'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL',
                 'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN',
                 'LOCATIONNAME_0', 'LOCATIONNAME_1', 'LOCATIONNAME_2',
                 'LOCATIONNAME_3', 'LOCATIONNAME_4', 'LOCATIONNAME_5', 'LOCATIONNAME_6',
                 'LOCATIONNAME_7', 'LOCATIONNAME_8', 'LOCATIONNAME_9', 'DISTRICTS_ARGANZUELA',
                 'DISTRICTS_BARAJAS', 'DISTRICTS_CARABANCHEL', 'DISTRICTS_CENTRO',
                 'DISTRICTS_CHAMARTIN', 'DISTRICTS_CHAMBERI', 'DISTRICTS_CIUDAD LINEAL',
                 'DISTRICTS_FUENCARRAL-EL PARDO', 'DISTRICTS_HORTALEZA', 'DISTRICTS_LATINA',
                 'DISTRICTS_MONCLOA-ARAVACA', 'DISTRICTS_MORATALAZ', 'DISTRICTS_PUENTE DE VALLECAS',
                 'DISTRICTS_RETIRO', 'DISTRICTS_SALAMANCA', 'DISTRICTS_SAN BLAS-CANILLEJAS',
                 'DISTRICTS_TETUAN', 'DISTRICTS_USERA', 'DISTRICTS_VICALVARO',
                 'DISTRICTS_VILLA DE VALLECAS', 'DISTRICTS_VILLAVERDE']]

    # Try with all the features do not give a good result
    # Group 3
    '''df_new = df[[
    "PRICE", "UNITPRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "ISPARKINGSPACEINCLUDEDINPRICE",
    "PARKINGSPACEPRICE", "HASNORTHORIENTATION", "HASSOUTHORIENTATION",
    "HASEASTORIENTATION", "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE",
    "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO",
    "ISINTOPFLOOR", "FLOORCLEAN", "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR",
    "CADDWELLINGCOUNT", "CADASTRALQUALITYID", "BUILTTYPEID_1", "BUILTTYPEID_2",
    "BUILTTYPEID_3", "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO",
    "DISTANCE_TO_CASTELLANA", "PERIOD_201803", "PERIOD_201806", "PERIOD_201809",
    "PERIOD_201812", "LOCATIONNAME_0", "LOCATIONNAME_1", "LOCATIONNAME_2",
    "LOCATIONNAME_3", "LOCATIONNAME_4", "LOCATIONNAME_5", "LOCATIONNAME_6",
    "LOCATIONNAME_7", "LOCATIONNAME_8", "LOCATIONNAME_9", 'DISTRICTS_ARGANZUELA',
    'DISTRICTS_BARAJAS', 'DISTRICTS_CARABANCHEL', 'DISTRICTS_CENTRO',
    'DISTRICTS_CHAMARTIN', 'DISTRICTS_CHAMBERI', 'DISTRICTS_CIUDAD LINEAL',
    'DISTRICTS_FUENCARRAL-EL PARDO', 'DISTRICTS_HORTALEZA', 'DISTRICTS_LATINA',
    'DISTRICTS_MONCLOA-ARAVACA', 'DISTRICTS_MORATALAZ', 'DISTRICTS_PUENTE DE VALLECAS',
    'DISTRICTS_RETIRO', 'DISTRICTS_SALAMANCA', 'DISTRICTS_SAN BLAS-CANILLEJAS',
    'DISTRICTS_TETUAN', 'DISTRICTS_USERA', 'DISTRICTS_VICALVARO',
    'DISTRICTS_VILLA DE VALLECAS', 'DISTRICTS_VILLAVERDE'
    ]]'''

    X = df_new.drop(['PRICE', 'UNITPRICE'], axis=1)
    y = df_new['PRICE']

    # Separation between frequent and not frequent classes
    frequent_classes = y.value_counts()[y.value_counts() > 1].index
    rare_classes = y.value_counts()[y.value_counts() == 1].index

    X_stratified = X.loc[y.isin(frequent_classes)]
    y_stratified = y.loc[y.isin(frequent_classes)]
    X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(
        X_stratified, y_stratified, test_size=test_size_fraction, stratify=y_stratified, random_state=42)

    X_rare = X.loc[y.isin(rare_classes)]
    y_rare = y.loc[y.isin(rare_classes)]
    X_train_rare, X_test_rare, y_train_rare, y_test_rare = train_test_split(
        X_rare, y_rare, test_size=test_size_fraction, random_state=42)

    # Concat frequent and not frequent classes
    X_train = pd.concat([X_train_st, X_train_rare])
    X_test = pd.concat([X_test_st, X_test_rare])
    y_train = pd.concat([y_train_st, y_train_rare])
    y_test = pd.concat([y_test_st, y_test_rare])

    return X_train, X_test, y_train, y_test