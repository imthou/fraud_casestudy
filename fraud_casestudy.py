import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from bs4 import BeautifulSoup
import cPickle as pickle

def num_of_tickets(data):
    return len(data)

def total_number_of_tickets_for_sale(data):
    total = 0
    for d in data:
        total += d['quantity_total']
    return total

def total_ticket_costs(data):
    total = 0
    for d in data:
        total+= d['quantity_total']*d['cost']
    return total

def number_of_tickets_sold(data):
    total = 0
    for d in data:
        total += d['quantity_sold']
    return total

def preprocessing(df):
    events = ['event_created','event_end','event_published','event_start','approx_payout_date']
    for event in events:
        df[event] = pd.to_datetime(df[event], unit='s')
    df['event_duration'] = (df['event_end'] - df['event_start']).dt.days
    df['event_pubtostart'] = (df['event_start'] - df['event_published']).dt.days
    fraud_accts = ['fraudster_event','fraudster','locked','tos_lock','fraudster_att']
    df['isfraud'] = df['acct_type'].apply(lambda x: x in fraud_accts)
    df['country'] = df['country'].replace("","No Entry").fillna('No Entry')
    df['listed'] = df.listed.apply(lambda x: x== 'y')
    df['email_fraud'] = zip(df.email_domain.tolist(),df.isfraud.tolist())
    df['email_fraud'] = df['email_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])
    # df['payee_fraud'] = zip(df['payee_name'].tolist(),df.isfraud.tolist())
    # df['payee_fraud'] = df['payee_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])


    '''
    Jennifer
    '''
    df['has_img'] = ['img' in r for r in df['description']]
    df['name'].replace('','NO NAME GIVEN', inplace=True)
    df['percent_caps']=[sum(1 for c in r if c.isupper())/float(len(r)) for r in df['name']]
    df['name_lessthantwo'] = [r < 2 for r in df['name_length']]
    df['nogts'] = [r == 0 for r in df['gts']]
    df['text_desc'] = [BeautifulSoup(desc, "lxml").get_text() for desc in df['description']]
    df['percent_exc'] = [sum(1 for c in r if c == '!')/float(sum(1 for c in r if c in ['.','?','!'])+1) for r in df['text_desc']]
    df['has_bar'] = ['bar' in r for r in df['text_desc']]
    df['has_city'] = ['city' in r for r in df['text_desc']]
    df['has_club'] = ['club' in r for r in df['text_desc']]
    df['has_dj'] = ['dj' in r for r in df['text_desc']]
    df['has_events'] = ['events' in r for r in df['text_desc']]
    df['has_live'] = ['live' in r for r in df['text_desc']]
    df['has_nyc'] = ['nyc' in r for r in df['text_desc']]
    df['has_open'] = ['open' in r for r in df['text_desc']]
    df['has_party'] = ['party' in r for r in df['text_desc']]
    df['has_place'] = ['place' in r for r in df['text_desc']]
    df['has_contact'] = ['contact' in r for r in df['text_desc']]
    df['has_group'] = ['group' in r for r in df['text_desc']]
    df['has_life'] = ['life' in r for r in df['text_desc']]
    df['has_registration'] = ['registration' in r for r in df['text_desc']]
    df['has_session'] = ['session' in r for r in df['text_desc']]
    df['has_social'] = ['social' in r for r in df['text_desc']]
    df['has_training'] = ['training' in r for r in df['text_desc']]
    df['has_work'] = ['work' in r for r in df['text_desc']]
    df['has_workshop'] = ['workshop' in r for r in df['text_desc']]

    '''
    Jesse
    '''
    df['has_orgdesc'] = df.org_desc.apply(lambda x: x != '')
    df['has_fbkcateg'] = df.org_facebook.apply(lambda x: x != 0)
    df.groupby('isfraud')['has_fbkcateg'].value_counts()
    df['has_twitteracctnum'] = df.org_twitter.apply(lambda x: x != 0)
    df['has_url'] = ['http' in r for r in df['org_desc']]
    df['payout_type'] = df['payout_type'].replace('','UNK')
    df['has_prev_payout'] = df['previous_payouts'].apply(lambda x: 0 if not x else 1)
    fraud_list = ['LIDF','Global Gas Card','Ultimate Wine','Pocket Pictures', 'FORD MODELS UK',
              'Rotary Club of East Los Angeles', 'Tree of Life', 'Startup Saturdays', 'Gametightny.com', 'Ger-Nis Culinary & Herb Center',
              'Joonbug', 'Market District Robinson', 'Premier Events', 'Pocket Pictures', 'STYLEPARTIES', 'Blow The Whistle On Bullying ~ It Matters What We do',
              'Network After Work','Museum of Contemporary Art, North Miami', "Mabs' Events", 'DC Black Theatre Festival', 'stephen',
              '1st Class Travel Club', 'ELITE SOCIAL', 'Absolution Chess Club']
    df['org_name_naughty_list'] = df['org_name'].isin(fraud_list)

    '''
    Muneeb
    '''
    fs = [num_of_tickets, total_number_of_tickets_for_sale, total_ticket_costs, number_of_tickets_sold] #
    ns = ['num_of_tickets', 'total_number_of_tickets_for_sale', 'total_ticket_costs', 'number_of_tickets_sold'] #
    for i,f in enumerate(fs):
        df[ns[i]] = df['ticket_types'].map(f)

    df = pd.get_dummies(df, columns=['country','currency', 'payout_type','has_header','user_type','email_fraud']) #
    dropped_columns = ['acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','email_fraud_NA'] #
    df_rf = df.drop(dropped_columns, axis=1)

    return df_rf

def grid_search_rf(X_train, X_test, y_train, y_test):

    random_forest_grid = {'n_estimators': [10],
                            'criterion': ['gini','entropy'],
                            'min_samples_split': [2, 4, 6],
                            'min_samples_leaf': [1, 2],
                            'n_jobs': [-1],
                            'max_features': ['sqrt',None,'log2'],
                            'class_weight': ['balanced']}

    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='accuracy')

    rf_gridsearch.fit(X_train, y_train)

    print "best parameters:", rf_gridsearch.best_params_

    best_rf_model = rf_gridsearch.best_estimator_

    y_pred = best_rf_model.predict(X_test)

    print "Accuracy with best rf:", cross_val_score(best_rf_model, X_test, y_test, scoring='accuracy').mean()

    rf = RandomForestClassifier(n_estimators=10, oob_score=True, class_weight='balanced')

    print "Accuracy with default param rf:", cross_val_score(rf, X_test, y_test, scoring='accuracy').mean()

    return best_rf_model, y_pred

    """
    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    7.1s
    [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed:   19.5s finished
    best parameters: {'n_jobs': -1, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 6, 'criterion': 'entropy', 'max_features': None, 'class_weight': 'balanced'}
    Accuracy with best rf: 0.980248256641
    Accuracy with default param rf: 0.97795067396
    1 has_prev_payout 0.486136855529
    2 org_name_naughty_list 0.0916096652899
    3 event_pubtostart 0.0626704618523
    4 sale_duration 0.0439038447356
    5 number_of_tickets_sold 0.0374632361626
    6 user_age 0.0348488521027
    7 venue_latitude 0.0282192955725
    8 num_payouts 0.0214216148838
    9 percent_caps 0.0208179935883
    10 name_length 0.0193603649767
    11 total_ticket_costs 0.0190708204624
    12 body_length 0.0160814330844
    13 payout_type_CHECK 0.0160700103712
    14 total_number_of_tickets_for_sale 0.0129293705687
    15 num_order 0.0125510322952
    16 num_of_tickets 0.0107134460471
    17 user_type_1 0.00930272951404

    -- Without has_prev_payout and number_of_tickets_sold

    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    7.7s
    [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed:   21.7s finished
    best parameters: {'n_jobs': -1, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 6, 'criterion': 'gini', 'max_features': 'sqrt', 'class_weight': 'balanced'}
    Accuracy with best rf: 0.965547809402
    Accuracy with default param rf: 0.964169133222
    1 num_payouts 0.135155301107
    2 sale_duration 0.094024490362
    3 user_age 0.0884867609544
    4 event_pubtostart 0.0712436592083
    5 num_order 0.0646571815758
    6 name_length 0.0371901533424
    7 user_type_1 0.0365332387827
    8 total_ticket_costs 0.0351809633672
    9 venue_longitude 0.0332945710792
    10 body_length 0.0308185133787
    11 percent_caps 0.0303091693283
    12 org_name_naughty_list 0.0301539356673
    13 payout_type_CHECK 0.0276021574641
    14 venue_latitude 0.0245117201326
    15 payout_type_UNK 0.019535031278
    16 total_number_of_tickets_for_sale 0.0162146810573
    17 delivery_method 0.0152312758367
    18 has_fbkcateg 0.0143702005143
    19 num_of_tickets 0.0142664896462
    20 percent_exc 0.013919633783
    21 nogts 0.0119602713044
    22 has_twitteracctnum 0.011742663146
    23 user_type_4 0.0114926326247
    24 user_type_3 0.00752269430185
    25 channels 0.00751110054057
    26 has_header_1.0 0.00728268603622
    27 has_orgdesc 0.0065489213142
    28 has_analytics 0.00628510914163
    29 payout_type_ACH 0.00594355687471
    30 has_url 0.00509880969317
    31 event_duration 0.00504007654125
    32 has_header_0.0 0.00500698049517
    33 has_work 0.00491629598962
    34 has_bar 0.00490308318809
    35 fb_published 0.00429750929033
    36 currency_AUD 0.00408252654253
    37 currency_USD 0.00383133808686
    38 has_place 0.00350121773782
    39 country_US 0.00338219630672
    40 has_party 0.00331897179514
    41 has_open 0.00324334921978
    42 has_life 0.00267993667201
    43 currency_GBP 0.00261184613952
    44 has_registration 0.00254728869632
    45 has_img 0.0025269810097
    46 country_CA 0.00229141315181
    47 country_AU 0.0022759428813
    48 has_contact 0.0020544068529
    49 has_logo 0.00196961545094
    50 has_club 0.00194378044194
    51 has_live 0.00186539674913
    52 has_city 0.00180009660407
    53 has_training 0.00162622967306
    54 currency_CAD 0.0015639705578
    55 has_workshop 0.00152289681671

    -- model with emails included:
    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   21.0s
    [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed:  1.0min finished
    best parameters: {'n_jobs': -1, 'min_samples_leaf': 1, 'n_estimators': 10, 'min_samples_split': 6, 'criterion': 'gini', 'max_features': None, 'class_weight': 'balanced'}
    Accuracy with best rf: 0.98667981108
    Accuracy with default param rf: 0.975654040562
    1 has_prev_payout 0.547772670553
    2 org_name_naughty_list 0.0935942928815
    3 email_fraud_gmail.com 0.0915884825363
    4 email_fraud_yahoo.com 0.057950695172
    5 number_of_tickets_sold 0.0411339675444
    6 sale_duration 0.0295281808699
    7 user_age 0.0086484597202
    8 email_fraud_hotmail.com 0.00669476668126
    9 event_pubtostart 0.00593280793825
    10 email_fraud_att.net 0.00565694082569
    11 payout_type_CHECK 0.00533290874594
    12 email_fraud_certifiedforensicloanauditors.com 0.00530906514231
    13 email_fraud_cs.com 0.00525573723153
    14 total_ticket_costs 0.00470077072966
    15 email_fraud_girl-geeks.co.uk 0.00453709283762
    16 venue_longitude 0.00412411441056
    17 email_fraud_msn.com 0.00398170064888
    18 num_payouts 0.00390112484863
    19 email_fraud_ymail.com 0.00374229164202
    20 email_fraud_yahoo.ca 0.00363756839217
    21 email_fraud_bravuraconsulting.com 0.00362930233024
    22 email_fraud_premier3.com 0.00350991454219
    23 email_fraud_datachieve.com 0.00343427745577
    24 email_fraud_theparadigmcollective.com 0.00307969359718
    25 email_fraud_socialsolutionsacademy.co.uk 0.00289260499674
    26 email_fraud_nantucketfilmfestival.org 0.00285096726047
    27 email_fraud_REDCLAYSOULNOLA.COM 0.00282459463776
    28 email_fraud_alhambrapalacerestaurant.com 0.00261351812457
    29 email_fraud_republicanflorida.com 0.00258113375694
    30 email_fraud_ctq.net.au 0.00256841122458
    31 venue_latitude 0.00240789629675
    32 email_fraud_heresyemail.com 0.00229163514074
    33 email_fraud_launchpadinw.com 0.00227025662073
    34 email_fraud_cox.net 0.00171377050507
    35 email_fraud_ladiesat11.com 0.0017031234941
    36 email_fraud_aol.com 0.00166137299293
    37 total_number_of_tickets_for_sale 0.00152031164581
    38 email_fraud_cbsdcalumclub.com 0.00149002279165
    39 email_fraud_yellamo.com 0.00145992664504
    40 num_order 0.00120214171041
    41 name_length 0.00119097803646
    42 email_fraud_pridetoronto.com 0.00116491091519
    43 email_fraud_live.com 0.00115375894818
    44 email_fraud_betaltd.org 0.00115046200527
    45 payout_type_ACH 0.00103579048193
    46 user_type_3 0.000935138996154
    47 percent_caps 0.000884365742757
    48 email_fraud_smokinbettys.com 0.000878378269438
    49 email_fraud_live.de 0.000869624146193

    """

def plot_confusion_matrix(y_test, y_pred, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = my_confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print 'Confusion matrix, without normalization'
    print cm
    plt.clf()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.grid([])
    plt.savefig('confusion_mat_fraud.png')

def my_confusion_matrix(y_test, y_pred):
    return pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True']).reindex_axis([1,0], axis=1).reindex_axis([1,0], axis=0)

def print_featimpt(df2, best_rf_model, percent=.99):
    lab_feats = sorted(zip(df2.columns, best_rf_model.feature_importances_), key=lambda x : x[1])[::-1]

    total,cnt = 0,0
    for n,v in lab_feats:
        total+=v
        if total<=percent:
            cnt+=1
            print cnt,n,v

def pickle_model(model, name):
    with open("{}.pkl".format(name), 'w') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    df = pd.read_json('data/train_new.json')
    df2 = preprocessing(df)
    df2.dropna(inplace=True)

    y = df2.pop('isfraud').values
    X = df2.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    best_rf_model, y_pred = grid_search_rf(X_train, X_test, y_train, y_test)

    print_featimpt(df2, best_rf_model)

    plot_confusion_matrix(y_test, y_pred, [1,0])

    pickle_model(best_rf_model, name='model3_w_email')
