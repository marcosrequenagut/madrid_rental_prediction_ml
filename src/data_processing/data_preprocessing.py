import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preproces_data(URL, test_size_fraction):# Load the data
    df = pd.read_csv(URL)

    # Select important feautres
    '''df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
                 'HASTERRACE', 'HASLIFT', 'ISPARKINGSPACEINCLUDEDINPRICE',
                 'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL', 'HASGARDEN',
                 'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN',
                 'PERIOD_201803', 'PERIOD_201806', 'PERIOD_201809', 'PERIOD_201812']]'''

    # These combination offers a better accuracy
    df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
                 'HASTERRACE', 'ISPARKINGSPACEINCLUDEDINPRICE',
                 'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL',
                 'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN']]

    # Try with all the features do not give a good result
    '''df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
'ROOMNUMBER', 'BATHNUMBER', 'HASTERRACE',
'HASLIFT', 'HASAIRCONDITIONING', 'ISPARKINGSPACEINCLUDEDINPRICE',
'PARKINGSPACEPRICE', 'HASNORTHORIENTATION', 'HASSOUTHORIENTATION',
'HASEASTORIENTATION', 'HASWESTORIENTATION', 'HASBOXROOM',
'HASWARDROBE', 'HASSWIMMINGPOOL', 'HASDOORMAN',
'HASGARDEN', 'ISDUPLEX', 'ISSTUDIO',
'ISINTOPFLOOR', 'FLOORCLEAN', 'CADCONSTRUCTIONYEAR',
'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 'CADASTRALQUALITYID',
'BUILTTYPEID_1', 'BUILTTYPEID_2', 'BUILTTYPEID_3',
'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA', 'YEAR',
'PERIOD_201803', 'PERIOD_201806',
'PERIOD_201809', 'PERIOD_201812', 'AMENITYID_1',
'AMENITYID_2', 'AMENITYID_3',]]'''

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