import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt        # For plotting graphs
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

#Load data from csv file
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
#Keep a copy of the original data
train_original=train.copy()
test_original=test.copy()

train_copy =train.copy()
test_copy = test.copy()

print train.describe(include="all")

print train.shape
#------Output------#
#(2009, 22)
#We have 2009 rows and 22 columns

print  "\n\n" , train.columns
#Display All The Column Name
#Index([u'uid', u'total_facebook_statuses', u'account_membership_period',
#       u'service_support_calls', u'number_of_snaps', u'total_whatsapp_charge',
#      u'sub_country_code', u'total_twitter_tweets', u'country',
#       u'total_email_characters', u'email_plan',
#       u'total_whatsapp_msg_characters', u'social_account_number',
#       u'total_whatsapp_msgs', u'total_facebook_charge',
#       u'total_twitter_charge', u'snapchat_plan', u'total_email_charge',
#      u'total_twitter_tweet_characters', u'total_emails',
#      u'total_facebook_status_characters', u'turn_anti_social'],
#     dtype='object')

print train['turn_anti_social'].value_counts()
#0    1720    means out of 2009 , 1720 have not turned anti social
#1     289    means out of 2009 , 289 have turned anti social
#Name: turn_anti_social, dtype: int64

print train['turn_anti_social'].value_counts(normalize=True)
# Output In proportion
#0    0.856147
#1    0.143853
#Name: turn_anti_social, dtype: float64

print train.dtypes
#Categorical variables in our dataset are: country , email_plan ,snapchat_plan , turn_anti_social,service_support_calls

plt.figure(1)
plt.subplot(131)
train['email_plan'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'email_plan')
#Around 85% do not have email plan
plt.subplot(132)
train['snapchat_plan'].value_counts(normalize=True).plot.bar(title= 'snapchat_plan')
#Around 78% do no have snapchat_plan
plt.subplot(133)
train['service_support_calls'].value_counts(normalize=True).plot.bar(title= 'service_support_calls')
#Most of the customer have made a single service_support_calls
plt.show()



plt.figure(1)
plt.subplot(221)
train['total_whatsapp_charge'].plot.box(figsize=(16,5))
plt.subplot(222)
train['total_twitter_charge'].plot.box(figsize=(16,5))
plt.subplot(223)
train['total_email_charge'].plot.box(figsize=(16,5))
plt.subplot(224)
train['total_facebook_charge'].plot.box(figsize=(16,5))

plt.show()

#We observe that there are not much outliers for the above columns

plt.figure(1)
plt.subplot(221)
train['total_whatsapp_msgs'].plot.box(figsize=(16,5))
plt.subplot(222)
train['total_twitter_tweets'].plot.box(figsize=(16,5))
plt.subplot(223)
train['total_emails'].plot.box(figsize=(16,5))
plt.subplot(224)
train['total_facebook_statuses'].plot.box(figsize=(16,5))

plt.show()


email_subscribe=pd.crosstab(train['email_plan'],train['turn_anti_social'])
email_subscribe.div(email_subscribe.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# Around 12% pepole who do not have email plan turns anti-social
# Around 45% people who have email plan turns anti-social
# means a person having email plan has higher chances of being anti social
plt.show()

snapchat_subscribe=pd.crosstab(train['snapchat_plan'],train['turn_anti_social'])
snapchat_subscribe.div(snapchat_subscribe.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# It can be inferred that the proportion of snapchat suscribers and non suscribers
# is more or less same for both being anti social and social
plt.show()

service_support_calls=pd.crosstab(train['service_support_calls'],train['turn_anti_social'])
service_support_calls.div(service_support_calls.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# It can be inferred that when service_support_calls is greater than 3
# chances of being anti social is more
plt.show()


country=pd.crosstab(train['country'],train['turn_anti_social'])
country.div(country.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# It can be inferred that almost all 50 country
# more or less same are both being anti social and social
plt.show()

df=train.dropna()

print
print train['total_facebook_statuses'].describe()
#Output we need
#count  2009.000000   , mean 100.637631
# min  30.000000 , max 160.000000
bins=[30,90,120,160]
group=['Low','Avg','High', ]
train['facebook_statuses_bin']=pd.cut(df['total_facebook_statuses'],bins,labels=group)
facebook_statuses_bin=pd.crosstab(train['facebook_statuses_bin'],train['turn_anti_social'])
facebook_statuses_bin.div(facebook_statuses_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('facebook_statuses')
P = plt.ylabel('Percentage')
plt.show()

print train['total_whatsapp_msgs'].describe()
#count    2009.000000   mean       99.970134  min        33.000000    max       175.000000
bins=[30,80,130,180]
group=['Low','Avg','High', ]
train['whatsapp_msgs_bin']=pd.cut(df['total_whatsapp_msgs'],bins,labels=group)
whatsapp_msgs_bin=pd.crosstab(train['whatsapp_msgs_bin'],train['turn_anti_social'])
whatsapp_msgs_bin.div(whatsapp_msgs_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('whatsapp_msgs')
P = plt.ylabel('Percentage')
plt.show()
print train['total_twitter_tweets'].describe()
#count    2009.000000  mean      100.481334  min         0.000000  max       170.000000
bins=[0,60,120,180]
group=['Low','Avg','High', ]
train['twitter_tweets_bin']=pd.cut(df['total_twitter_tweets'],bins,labels=group)
twitter_tweets_bin=pd.crosstab(train['twitter_tweets_bin'],train['turn_anti_social'])
twitter_tweets_bin.div(twitter_tweets_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('twitter_tweets')
P = plt.ylabel('Percentage')
plt.show()

print train['total_emails'].describe()
#count    2009.000000  mean        4.448980   min         0.000000   max        18.000000
bins=[0,4,10,18]
group=['Low','Avg','High', ]
train['total_emails_bin']=pd.cut(df['total_emails'],bins,labels=group)
total_emails_bin=pd.crosstab(train['total_emails_bin'],train['turn_anti_social'])
total_emails_bin.div(total_emails_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('total_emails')
P = plt.ylabel('Percentage')
plt.show()

print train['number_of_snaps'].describe()
#count    2009.000000  mean 7.989547   min 0.000000  max 50.000000
bins=[0,4,10,18]
group=['Low','Avg','High', ]
train['number_of_snaps_bin']=pd.cut(df['number_of_snaps'],bins,labels=group)
number_of_snaps_bin=pd.crosstab(train['number_of_snaps_bin'],train['turn_anti_social'])
number_of_snaps_bin.div(number_of_snaps_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('number_of_snaps')
P = plt.ylabel('Percentage')
plt.show()

train['total_facebook_charge'].describe()
#count    2009.000000  mean       30.684803  min         0.440000  max        58.960000
bins=[0,20,40,60]
group=['Low','Avg','High', ]
train['facebook_charge_bin']=pd.cut(df['total_facebook_charge'],bins,labels=group)
facebook_charge_bin=pd.crosstab(train['facebook_charge_bin'],train['turn_anti_social'])
facebook_charge_bin.div(facebook_charge_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('facebook_charge')
P = plt.ylabel('Percentage')
plt.show()
# As facebook charge increases more pepole turned anti social

print train['total_whatsapp_charge'].describe()
#count    1814.000000  mean        9.020138  min         1.040000   max        17.770000
bins=[0,6,12,18]
group=['Low','Avg','High', ]
train['whatsapp_charge_bin']=pd.cut(df['total_whatsapp_charge'],bins,labels=group)
whatsapp_charge_bin=pd.crosstab(train['whatsapp_charge_bin'],train['turn_anti_social'])
whatsapp_charge_bin.div(whatsapp_charge_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('whatsapp_charge')
P = plt.ylabel('Percentage')
plt.show()
# As facebook charge increases more pepole turned anti social but less than that of facebook
print train['total_twitter_charge'].describe()
#count    2009.000000   mean       17.105555  min         0.000000   max        30.910000
bins=[0,10,20,35]
group=['Low','Avg','High', ]
train['twitter_charge_bin']=pd.cut(df['total_twitter_charge'],bins,labels=group)
twitter_charge_bin=pd.crosstab(train['twitter_charge_bin'],train['turn_anti_social'])
twitter_charge_bin.div(twitter_charge_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('twitter_charge')
P = plt.ylabel('Percentage')
plt.show()

print train['total_email_charge'].describe()
#count    2009.000000   mean        2.766272   min         0.000000   max         5.400000
bins=[0,2,4,6]
group=['Low','Avg','High', ]
train['emails_charge_bin']=pd.cut(df['total_email_charge'],bins,labels=group)
emails_charge_bin=pd.crosstab(train['emails_charge_bin'],train['turn_anti_social'])
emails_charge_bin.div(emails_charge_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('emails_charge')
P = plt.ylabel('Percentage')
plt.show()

train['Total_charges']=train['total_facebook_charge'] + train['total_whatsapp_charge'] + train['total_twitter_charge'] + train['total_email_charge']
df=train.dropna()
print train['Total_charges'].describe()
#count    1814.000000  mean       59.536632  min        27.020000  max        96.150000
bins=[0,30,70,100]
group=['Low','Avg','High', ]
train['Total_charges_bin']=pd.cut(df['Total_charges'],bins,labels=group)
Total_charges_bin=pd.crosstab(train['Total_charges_bin'],train['turn_anti_social'])
Total_charges_bin.div(Total_charges_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_charges')
P = plt.ylabel('Percentage')
plt.show()


train_original['email_plan'].replace('yes', 1,inplace=True)
train_original['email_plan'].replace('no', 0,inplace=True)
train_original['snapchat_plan'].replace('yes', 1,inplace=True)
train_original['snapchat_plan'].replace('no', 0,inplace=True)
train_original=train_original.drop(['uid','country', 'sub_country_code', 'social_account_number', 'account_membership_period'], axis=1)
test= test.drop(['uid','country', 'sub_country_code', 'social_account_number', 'account_membership_period'], axis=1)

matrix = train_original.corr()
sns.heatmap(matrix, vmax=1, square=True, cmap="BuPu");
plt.subplots_adjust(bottom=0.4)
plt.show()
##################################################################
# The columns which are highly correlated are shown as
# dark square box in the graph.
#################################################################


#-------Missing values-------------#

train_original.isnull().sum()
#total_whatsapp_charge    195
#total_email_characters   201
#total_twitter_tweet_characters      195
#total_facebook_status_characters    198


train_original['total_whatsapp_charge'].fillna(train_original['total_whatsapp_charge'].median(), inplace=True)
train_original['total_email_characters'].fillna(train_original['total_email_characters'].median(), inplace=True)
train_original['total_twitter_tweet_characters'].fillna(train_original['total_twitter_tweet_characters'].median(), inplace=True)
train_original['total_facebook_status_characters'].fillna(train_original['total_facebook_status_characters'].median(), inplace=True)

print train_original.isnull().sum()

test.isnull().sum()
#total_whatsapp_charge               17
#total_email_characters              16
#total_twitter_tweet_characters      19
#total_facebook_status_characters    17

test['total_whatsapp_charge'].fillna(test['total_whatsapp_charge'].median(), inplace=True)
test['total_email_characters'].fillna(test['total_email_characters'].median(), inplace=True)
test['total_twitter_tweet_characters'].fillna(test['total_twitter_tweet_characters'].median(), inplace=True)
test['total_facebook_status_characters'].fillna(test['total_facebook_status_characters'].median(), inplace=True)
test.isnull().sum()

print train_original.columns
print train_original.shape


train_copy=train_original
test_copy=test
x = train_original.drop('turn_anti_social',1)
y = train.turn_anti_social
x=pd.get_dummies(x)
test['email_plan'].replace('yes', 1,inplace=True)
test['email_plan'].replace('no', 0,inplace=True)
test['snapchat_plan'].replace('yes', 1,inplace=True)
test['snapchat_plan'].replace('no', 0,inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x, y,
                            test_size = 0.20, random_state = 0)


from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg
#MODEL-1: Accuracy of LogisticRegression :  85.82

#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian
#MODEL-2: Accuracy of GaussianNB :  84.83

#MODEL-3) Support Vector Machines
#------------------------------------------
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-3: Accuracy of Support Vector Machines : ", acc_svc

#OUTPUT:-
#MODEL-3: Accuracy of Support Vector Machines :  85.57

#MODEL-4) Linear SVC
#------------------------------------------
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc

#OUTPUT:-
#MODEL-4: Accuracy of LinearSVC :  20.15

#MODEL-5) Perceptron
#------------------------------------------
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-5: Accuracy of Perceptron : ",acc_perceptron

#OUTPUT:-
#MODEL-5: Accuracy of Perceptron :  85.57


#MODEL-6) Decision Tree Classifier
#------------------------------------------
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree

#OUTPUT:-
#MODEL-6: Accuracy of DecisionTreeClassifier :  91.29


#MODEL-7) Random Forest
#------------------------------------------
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest

#OUTPUT:-
#MODEL-7: Accuracy of RandomForestClassifier :  95.02

#MODEL-8) KNN or k-Nearest Neighbors
#------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn

#OUTPUT:-
#MODEL-8: Accuracy of k-Nearest Neighbors :  86.57

#MODEL-9) Stochastic Gradient Descent
#------------------------------------------
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd

#OUTPUT:-
#MODEL-9: Accuracy of Stochastic Gradient Descent :  85.57


#MODEL-10) Gradient Boosting Classifier
#------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk

#OUTPUT:-
#MODEL-10: Accuracy of GradientBoostingClassifier :  95.52

#Model used is DecisionTree whose accuracy is 91.29
prediction_decisiontree = decisiontree.predict(test)
output_decisiontree = pd.DataFrame({'turn_anti_social': prediction_decisiontree })
output_decisiontree.to_csv('decisiontree.csv', index=False) #index=false means no row heading

#Model used is randomforestclassifier whose accuracy is 94.78
prediction_randomforest = randomforest.predict(test)
output_randomforest = pd.DataFrame({'turn_anti_social': prediction_randomforest })
output_randomforest.to_csv('randomforest.csv', index=False) #index=false means no row heading


#Model used is GradientBoostingClassifier whose accuracy is 97.51
prediction = gbk.predict(test)
output = pd.DataFrame({'turn_anti_social': prediction })
output.to_csv('gradientboosting.csv', index=False) #index=false means no row heading
print output['turn_anti_social'].value_counts(normalize=True)






###########################################################################################################
#              INTRODUCTION OF NEW FEATURE : FEATURE ENGINEERING                                          #
#                                                                                                         #
#          NEW FEATURE ADDED IS: TOTAL CHARGES (SUM OF ALL THE CHARGES)                                   #
#                                                                                                         #
# TOTAL CHARGES=total_whatsapp_charge +  total_facebook_charge  + total_twitter_charge +total_email_charge#
###########################################################################################################



##################################################################################
#
# Add a column i.e Total_charges
# drop the column 1.total_whatsapp_charge
#                 2.total_facebook_charge
#                 3.total_twitter_charge
#                 4.total_email_charge
#
###################################################################################
train_copy['Total_charges']=train_copy['total_facebook_charge'] + train_copy['total_whatsapp_charge'] + train_copy['total_twitter_charge'] + train_copy['total_email_charge']
test_copy['Total_charges']=test_copy['total_facebook_charge'] + test_copy['total_whatsapp_charge'] + test_copy['total_twitter_charge'] + test_copy['total_email_charge']

train_copy=train_copy.drop(['total_whatsapp_charge', 'total_facebook_charge', 'total_twitter_charge', 'total_email_charge'], axis=1)
test_copy= test_copy.drop(['total_whatsapp_charge', 'total_facebook_charge', 'total_twitter_charge', 'total_email_charge'], axis=1)

x = train_copy.drop('turn_anti_social',1)
x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x, y,
                            test_size = 0.20, random_state = 0)
print
print
print "ACCURACY AFTER IMPLEMENTATION OF FEATURE ENGINEERING"
print
print
#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg
#MODEL-1: Accuracy of LogisticRegression :  86.32

#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian
#MODEL-2: Accuracy of GaussianNB : 86.32

#MODEL-3) Support Vector Machines
#------------------------------------------
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-3: Accuracy of Support Vector Machines : ", acc_svc

#OUTPUT:-
#MODEL-3: Accuracy of Support Vector Machines :  85.57



#MODEL-4) Linear SVC
#------------------------------------------
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc

#OUTPUT:-
#MODEL-4: Accuracy of LinearSVC :  85.57




#MODEL-5) Perceptron
#------------------------------------------
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-5: Accuracy of Perceptron : ",acc_perceptron

#OUTPUT:-
#MODEL-5: Accuracy of Perceptron :  85.57




#MODEL-6) Decision Tree Classifier
#------------------------------------------
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree

#OUTPUT:-
#MODEL-6: Accuracy of DecisionTreeClassifier :  96.02





#MODEL-7) Random Forest
#------------------------------------------
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest

#OUTPUT:-
#MODEL-7: Accuracy of RandomForestClassifier :  94.78





#MODEL-8) KNN or k-Nearest Neighbors
#------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn

#OUTPUT:-
#MODEL-8: Accuracy of k-Nearest Neighbors :  86.57







#MODEL-9) Stochastic Gradient Descent
#------------------------------------------
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd

#OUTPUT:-
#MODEL-9: Accuracy of Stochastic Gradient Descent :  84.59




#MODEL-10) Gradient Boosting Classifier
#------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk

#OUTPUT:-
#MODEL-10: Accuracy of GradientBoostingClassifier :  97.51


################################################################################
#
#                 CREATING OUTPUT RESULT CSV FILE
# we need create a submission.csv file which includes our predictions for test data
#
################################################################################



test_copy['email_plan'].replace('yes', 1,inplace=True)
test_copy['email_plan'].replace('no', 0,inplace=True)
test_copy['snapchat_plan'].replace('yes', 1,inplace=True)
test_copy['snapchat_plan'].replace('no', 0,inplace=True)
#Model used is Decision Tree whose accuracy is 96.02
prediction_decisiontree = decisiontree.predict(test_copy)
output_decisiontree = pd.DataFrame({'turn_anti_social': prediction_decisiontree })
output_decisiontree.to_csv('decisiontree_feature.csv', index=False) #index=false means no row heading

#Model used is randomforestclassifier whose accuracy is 94.78
prediction_randomforest = randomforest.predict(test_copy)
output_randomforest = pd.DataFrame({'turn_anti_social': prediction_randomforest })
output_randomforest.to_csv('randomforest_feature.csv', index=False) #index=false means no row heading


#Model used is GradientBoostingClassifier whose accuracy is 97.51
prediction = gbk.predict(test_copy)
output = pd.DataFrame({'turn_anti_social': prediction })
output.to_csv('gradientboosting_feature.csv', index=False) #index=false means no row heading



print "All survival predictions done."
print "All predictions exported to submission.csv file."

