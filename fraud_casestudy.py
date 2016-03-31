import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from bs4 import BeautifulSoup
import cPickle as pickle

class FraudModel(object):
    """
    Preprocess dataset from e-commerce website
    Determine the best features to keep and engineer
    Build several models to use to detect fraudulent activity
    """

    def __init__(self):
        """
        Input: JSON Document
        Output: Dataframe

        Read in JSON document as Dataframe
        """
        self.df = pd.read_json('data/train_new.json')

    def preprocess_data(self):
        """
        Input: Dataframe
        Output: Dataframe

        Preprocessing steps:
        1. converted all event times to datetime variables and took the difference as features
        2. set all account types under fraud to be fraud for prediction
        3. converted email domains to categorical variable to use as features

        4. created boolean for images detected in description
        5. created feature to detect percent of company names in capitalization
        6. created feature to detect name length less than 2
        7. created variable to see if event has "gts" (grand total sales?)
        8. created feature to detect exclamation marks in text description
        9. created several features to detect if text description has the following: bar, city, club, dj, event, nyc, open, party, place, contact, group, life, registration, session, social, training, work, workshop

        10. created booleans to detect if event has organization description, facebook page, twitter account
        11. created feature to detect if description has url
        12. created boolean to detect if event has previously paid out
        13. created feature to detect if organization name is under the umbrella of a fraudulent list of companies

        14. extracted information on number of tickets for the event
        15. extracted information on total number of tickets for sale
        16. extracted information on total ticket costs
        17. extracted information on number of tickets sold

        18. dummified several categorical variables: 'country','currency', 'payout_type','has_header','user_type'
        19. dropped unimportant features: 'acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','email_fraud_NA'
        """
        events = ['event_created','event_end','event_published','event_start','approx_payout_date']
        for event in events:
            self.df[event] = pd.to_datetime(self.df[event], unit='s')
        self.df['event_duration'] = (self.df['event_end'] - self.df['event_start']).dt.days
        self.df['event_pubtostart'] = (self.df['event_start'] - self.df['event_published']).dt.days
        fraud_accts = ['fraudster_event','fraudster','locked','tos_lock','fraudster_att']
        self.df['isfraud'] = self.df['acct_type'].apply(lambda x: x in fraud_accts)
        self.df['country'] = self.df['country'].replace("","No Entry").fillna('No Entry')
        self.df['listed'] = self.df.listed.apply(lambda x: x== 'y')
        self.df['email_fraud'] = zip(self.df.email_domain.tolist(),self.df.isfraud.tolist())
        self.df['email_fraud'] = self.df['email_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])
        # self.df['payee_fraud'] = zip(self.df['payee_name'].tolist(),self.df.isfraud.tolist())
        # self.df['payee_fraud'] = self.df['payee_fraud'].apply(lambda x: x[0].replace(x[0],"NA") if x[1] == False else x[0])


        '''
        Jennifer
        '''
        self.df['has_img'] = ['img' in r for r in self.df['description']]
        self.df['name'].replace('','NO NAME GIVEN', inplace=True)
        self.df['percent_caps']=[sum(1 for c in r if c.isupper())/float(len(r)) for r in self.df['name']]
        self.df['name_lessthantwo'] = [r < 2 for r in self.df['name_length']]
        self.df['nogts'] = [r == 0 for r in self.df['gts']]
        self.df['text_desc'] = [BeautifulSoup(desc, "lxml").get_text() for desc in self.df['description']]
        self.df['percent_exc'] = [sum(1 for c in r if c == '!')/float(sum(1 for c in r if c in ['.','?','!'])+1) for r in self.df['text_desc']]
        self.df['has_bar'] = ['bar' in r for r in self.df['text_desc']]
        self.df['has_city'] = ['city' in r for r in self.df['text_desc']]
        self.df['has_club'] = ['club' in r for r in self.df['text_desc']]
        self.df['has_dj'] = ['dj' in r for r in self.df['text_desc']]
        self.df['has_events'] = ['events' in r for r in self.df['text_desc']]
        self.df['has_live'] = ['live' in r for r in self.df['text_desc']]
        self.df['has_nyc'] = ['nyc' in r for r in self.df['text_desc']]
        self.df['has_open'] = ['open' in r for r in self.df['text_desc']]
        self.df['has_party'] = ['party' in r for r in self.df['text_desc']]
        self.df['has_place'] = ['place' in r for r in self.df['text_desc']]
        self.df['has_contact'] = ['contact' in r for r in self.df['text_desc']]
        self.df['has_group'] = ['group' in r for r in self.df['text_desc']]
        self.df['has_life'] = ['life' in r for r in self.df['text_desc']]
        self.df['has_registration'] = ['registration' in r for r in self.df['text_desc']]
        self.df['has_session'] = ['session' in r for r in self.df['text_desc']]
        self.df['has_social'] = ['social' in r for r in self.df['text_desc']]
        self.df['has_training'] = ['training' in r for r in self.df['text_desc']]
        self.df['has_work'] = ['work' in r for r in self.df['text_desc']]
        self.df['has_workshop'] = ['workshop' in r for r in self.df['text_desc']]

        '''
        Jesse
        '''
        self.df['has_orgdesc'] = self.df.org_desc.apply(lambda x: x != '')
        self.df['has_fbkcateg'] = self.df.org_facebook.apply(lambda x: x != 0)
        self.df['has_twitteracctnum'] = self.df.org_twitter.apply(lambda x: x != 0)
        self.df['has_url'] = ['http' in r for r in self.df['org_desc']]
        self.df['payout_type'] = self.df['payout_type'].replace('','UNK')
        self.df['has_prev_payout'] = self.df['previous_payouts'].apply(lambda x: 0 if not x else 1)
        fraud_list = ['LIself.df','Global Gas Card','Ultimate Wine','Pocket Pictures', 'FORD MODELS UK',
                  'Rotary Club of East Los Angeles', 'Tree of Life', 'Startup Saturdays', 'Gametightny.com', 'Ger-Nis Culinary & Herb Center',
                  'Joonbug', 'Market District Robinson', 'Premier Events', 'Pocket Pictures', 'STYLEPARTIES', 'Blow The Whistle On Bullying ~ It Matters What We do',
                  'Network After Work','Museum of Contemporary Art, North Miami', "Mabs' Events", 'DC Black Theatre Festival', 'stephen',
                  '1st Class Travel Club', 'ELITE SOCIAL', 'Absolution Chess Club']
        self.df['org_name_naughty_list'] = self.df['org_name'].isin(fraud_list)

        '''
        Muneeb
        '''
        fs = [self.get_num_of_tickets, self.get_total_number_of_tickets_for_sale, self.get_total_ticket_costs, self.get_number_of_tickets_sold] #
        ns = ['num_of_tickets', 'total_number_of_tickets_for_sale', 'total_ticket_costs', 'number_of_tickets_sold'] #
        for i,f in enumerate(fs):
            self.df[ns[i]] = self.df['ticket_types'].map(f)


        self.df = pd.get_dummies(self.df, columns=['country','currency', 'payout_type','has_header','user_type','email_fraud']) #
        dropped_columns = ['acct_type','approx_payout_date','description','email_domain','event_created','event_end','event_published','event_start','name','object_id','org_desc','org_name','org_facebook','org_twitter','previous_payouts','sale_duration2','show_map','ticket_types','user_created','venue_address','venue_country','venue_name','venue_state','listed','gts','text_desc','payee_name','email_fraud_NA'] #
        self.df_rf = self.df.drop(dropped_columns, axis=1)

        return self.df_rf

    def get_num_of_tickets(self, ticket_types):
        """
        Output: Int
        """
        return len(ticket_types)

    def get_total_number_of_tickets_for_sale(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_total']
        return total

    def get_total_ticket_costs(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_total'] * d['cost']
        return total

    def get_number_of_tickets_sold(self, ticket_types):
        """
        Output: Int
        """
        total = 0
        for d in ticket_types:
            total += d['quantity_sold']
        return total


    def perform_grid_search_rf(self, X_train, X_test, y_train, y_test):
        """
        Output: Best model

        Perform grid search on all parameters of models to find the model that performs the best through cross-validation
        """

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


    def plot_confusion_matrix(self, y_test, y_pred, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        Output: Confusion Matrix Plot

        Plot confusion matrix to determine the number of false positive and false negatives produced from our model
        """
        cm = self.create_confusion_matrix(y_test, y_pred)
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

    def create_confusion_matrix(self, y_test, y_pred):
        """
        Output: Confusion matrix with appropriate labels
        """
        return pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True']).reindex_axis([1,0], axis=1).reindex_axis([1,0], axis=0)

    def print_featimpt(self, df2, best_rf_model, percent=.99):
        """
        Output: Feature Importances from Random Forest
        """
        lab_feats = sorted(zip(df2.columns, best_rf_model.feature_importances_), key=lambda x : x[1])[::-1]

        total,cnt = 0,0
        for n,v in lab_feats:
            total+=v
            if total<=percent:
                cnt+=1
                print cnt,n,v

    def pickle_model(self, model, name):
        """
        Output: Saved Model

        Pickles our model for later use
        """
        with open("{}.pkl".format(name), 'w') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    fm = FraudModel()
    df2 = fm.preprocess_data()
    df2.dropna(inplace=True)

    y = df2.pop('isfraud').values
    X = df2.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    best_rf_model, y_pred = fm.perform_grid_search_rf(X_train, X_test, y_train, y_test)

    fm.print_featimpt(df2, best_rf_model)

    fm.plot_confusion_matrix(y_test, y_pred, [1,0])

    # fm.pickle_model(best_rf_model, name='model3_w_email')
