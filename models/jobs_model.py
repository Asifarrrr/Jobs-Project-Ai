import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#=========================================== Data Cleaning =================================================

# Load the raw dataset
df = None

try:
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\data\data_science_job_posts_2025.csv")

except FileNotFoundError:
    pass


# Drop columns that are not used: location and headquarter have too many different values,
# and skills is stored as a text list that is hard to use directly
df = df.drop(columns = ['location','headquarter','skills'])


# Convert every column to text, remove extra spaces and make everything lowercase
# so the same value is always written the same way
columns = df.columns
for col in columns:
    df[col] = df[col].astype('string').str.strip().str.lower()


# Postings without a work status (remote / hybrid / on-site) get their own category
df['status'] = df['status'].fillna('unknown')


# "company_003" -> 3 (keep only the company number)
df['company'] = df['company'].str.replace('company_','')
df['company'] = pd.to_numeric(df['company'],errors = 'coerce')


# post_date looks like "17 days ago" or "a month ago".
# Split it into a number and a unit, then convert everything to days.
df['post_date'] = df['post_date'].str.replace('ago','').str.strip()
df[['number','unit']] = df['post_date'].str.split(' ',expand = True)
df['number'] = df['number'].str.strip()
df['unit'] = df['unit'].str.strip()

df = df.drop(columns = 'post_date')

df.loc[df['number'] == 'a','number'] = '1'                      # "a month ago" means 1 month
df['number'] = pd.to_numeric(df['number'],errors = 'coerce')
df.loc[(df['unit'] == 'month') | (df['unit'] == 'months') , 'number'] = df['number'] * 30     # months -> days
df.loc[(df['unit'] == 'year') | (df['unit'] == 'years') , 'number'] = df['number'] * 365      # years -> days
df.loc[df['unit'] == 'hours' , 'number'] = 0                    # posted today

df = df.drop(columns = 'unit')

# Move the new days column to position 4 (where post_date used to be)
col = df.pop('number')
df.insert(4,'post_date_days',col)


# Fill the few missing values in these columns with the most frequent value
for col in ['job_title','seniority_level','ownership']:
    df[col] = df[col].fillna(df[col].mode().iloc[0])


# Some rows have the word "private" in the company_size column, which really belongs to ownership
df.loc[df['company_size'] == 'private','ownership'] = 'private'


# Clean company_size: remove symbols and commas, convert to a number,
# and fill the values that could not be converted with the average size
df['company_size'] = (df['company_size'].str.replace('€','')
                                        .str.replace('b','')
                                        .str.replace(',','')
                                        .str.strip())
df['company_size'] = pd.to_numeric(df['company_size'],errors = 'coerce').astype('Int64')
df['company_size'] = df['company_size'].fillna(df['company_size'].mean().astype(int))


# Clean salary: it is either a range ("100000 - 200000") or a single value.
# Split it into a low and a high value; when there is only one value, use it for both.
df['salary'] = df['salary'].str.replace('€','').str.replace(',','').str.strip()
df[['salary1','salary2']] = df['salary'].str.split(' - ',expand = True)
df['salary1'] = pd.to_numeric(df['salary1'].str.strip(),errors = 'coerce')
df['salary2'] = pd.to_numeric(df['salary2'].str.strip(),errors = 'coerce')

df.loc[df['salary2'].isna(),'salary2'] = df['salary1']

# The final salary is the middle of the range
df['salary'] = ((df['salary1'] + df['salary2']) / 2).astype(int)
# Drop the helper columns and revenue (it mixes money values with words like "public" and "private")
df = df.drop(columns = ['salary1','salary2','revenue'])


# Target: split the salaries in two equal halves at the median
# 0 = lower half ("mid to bad rate"), 1 = upper half ("good rate")
df['salary_category'] = pd.qcut(
    df['salary'],
    q = 2,
    labels = ['low','high']
)

df['salary_category'] = df['salary_category'].cat.codes


# One-hot encode the text columns (one 0/1 column per category)
df = pd.get_dummies(df,columns = ['job_title','seniority_level','status'
                                 ,'industry','ownership'],dtype = int)

# Rename the columns so they match the names the API expects (no spaces or dashes)
df = df.rename(
    columns = {
        'job_title_data analyst': 'job_title_data_analyst',
        'job_title_data engineer': 'job_title_data_engineer',
        'job_title_data scientist': 'job_title_data_scientist',
        'job_title_machine learning engineer': 'job_title_machine_learning_engineer',
        'status_on-site': 'status_on_site',
    }
)

#=========================================== Ai model =================================================

# Features (x) and target (y). The salary itself is removed from the features
# because the target is made from it, and keeping it would leak the answer.
x = df.drop(columns = ['salary','salary_category'])
y = df['salary_category']

# Keep 18% of the data aside to test the model on rows it has never seen
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.18,
    random_state = 42,
)

# Random Forest: many decision trees voting together.
# max_depth, min_samples_leaf and min_samples_split limit the tree size to reduce overfitting.
model = RandomForestClassifier(
    n_estimators = 300,
    max_depth = 15,
    min_samples_leaf = 3,
    min_samples_split = 10,
    class_weight = 'balanced',
    random_state = 42
)
model.fit(x_train,y_train)

# Predict on the test rows (add accuracy / F1 here to evaluate the model)
y_pred = model.predict(x_test)

# Save the files used by the API and the dashboard
joblib.dump(model,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\data\model.pkl")
joblib.dump(df,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\models\data_frame.pkl")
joblib.dump(list(x.columns),r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\models\model_columns.pkl")