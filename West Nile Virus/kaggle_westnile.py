


import pandas as pd, numpy as np
import os
from datetime import date, datetime
import seaborn as sns

from bokeh.io import show
from bokeh.layouts import column
from bokeh.io import output_file, show, reset_output
from bokeh.models import HoverTool, ColumnDataSource, RangeTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral5


import h2o

#from sklearn.metrics import


pd.options.display.max_rows = 100
pd.options.display.max_columns = 20

os.getcwd()
proj = '/Users/zacklarsen/Desktop/Projects/Kaggle/West Nile'
#proj = 'C:\\Users\\U383387\\kaggle_projects\\westnile'
os.chdir(proj)
os.getcwd()
ls


# Load data
weather = pd.read_csv(proj + '/weather.csv')
train = pd.read_csv(proj + '/train.csv')
spray = pd.read_csv(proj + '/spray.csv')
test = pd.read_csv(proj + '/test.csv')

# Take a peek
weather.head()
train.head()
spray.head()



# Break weather up by station
station1 = weather[weather['Station'] == 1]
station2 = weather[weather['Station'] == 2]
station1.head()
station2.head()


# Merge data by day
merge1 = pd.merge(train, station1, on='Date')
merge1.head()

train_complete = pd.merge(merge1, station2, on='Date')
train_complete.head()

#train.set_index('Date').join(weather.set_index('Date'))
#pd.merge(train, weather, left_on='Date', right_on='Date')






# NA check
# https://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe

train_complete.apply(lambda x: sum(x.isnull().values), axis = 0) # For columns
train_complete.apply(lambda x: sum(x.isnull().values), axis = 1) # For rows

# Number of rows with at least one missing value
sum(train_complete.apply(lambda x: sum(x.isnull().values), axis = 1)>0)

train_complete.isnull().sum().sum() # Rows AND columns
train_complete.isnull().any(axis=1).sum() # Rows
train_complete.isnull().any(axis=0).sum() # Columns

np.count_nonzero(train_complete.isnull())
np.count_nonzero(train_complete.isnull().values.ravel())

train_complete.isnull().sum(axis=0)
train_complete.isnull().sum(axis=1)

# Display rows with at least one missing value
train_complete[train_complete.isnull().any(axis=1)]






# Descriptive statistics
train_complete.head()
train_complete.shape
train_complete.dtypes
train_complete.describe()
np.transpose(train_complete.describe())

train_complete.corr()
train_complete.cov()
train_complete['Block'].mean()
train_complete['AddressAccuracy'].describe(percentiles=[0.6,0.7,0.8,0.9])
train_complete['Block'].count()
train_complete['Block'].unique()
train_complete['NumMosquitos'].median()
train_complete['NumMosquitos'].mode()

# Interquartile range
train_complete['NumMosquitos'].quantile(0.75) - train_complete['NumMosquitos'].quantile(0.25)

train_complete.skew()

train_complete.kurtosis()





# Counts of categorical variable values/ pivot tables
train_complete.groupby('Species').size().sort_values(ascending=False)

train_complete.groupby(['WnvPresent','Species']).NumMosquitos.count()

train_complete.groupby(['Trap','Species']).NumMosquitos.count()











########################################
# Plotting
########################################


########################################
# Heatmap of correlations
########################################

sns.heatmap(train.corr())

sns.countplot(x=train["Species"])

sns.barplot(x="Species", y="NumMosquitos", data=train, ci=False)
sns.barplot(x="Species", y="NumMosquitos", data=train)

sns.factorplot(x="Species", y="NumMosquitos", data=train,
               hue="WnvPresent", col="WnvPresent", kind="bar", ci=False)





########################################
# Hexbin
########################################

n = len(train_complete)
x = train_complete['Longitude']
y = train_complete['Latitude']

p = figure(title="Hexbin for spray locations", match_aspect=True,
           tools="wheel_zoom,reset", background_fill_color='#440154')
p.grid.visible = False
r, bins = p.hexbin(x, y, size=0.025, hover_color="pink", hover_alpha=0.8)
p.circle(x, y, color="white", size=2)
p.add_tools(HoverTool(
    tooltips=[("count", "@c"), ("(lat,lon)", "(@Latitude, @Longitude)")],
    mode="mouse", point_policy="follow_mouse", renderers=[r]
))

output_file("hexbin.html")
show(p)


reset_output()
rm hexbin.html




########################################
#  Scatter plot
########################################

scatterDF = train[['Species','Trap','Latitude','Longitude','NumMosquitos','WnvPresent']]
scatterDF.shape # 10506, 6
scatterDF.head()
list(scatterDF.Species.unique())

colormap = {'CULEX PIPIENS/RESTUANS': 'red',
            'CULEX RESTUANS': 'green',
            'CULEX PIPIENS': 'blue',
            'CULEX SALINARIUS': 'purple',
            'CULEX TERRITANS': 'black',
            'CULEX TARSALIS': 'yellow',
            'CULEX ERRATICUS': 'orange'}

colors = [colormap[x] for x in scatterDF['Species']]
#colors[:10]

p = figure(title = "Skeeters by species")
p.xaxis.axis_label = 'Longitude'
p.yaxis.axis_label = 'Latitude'

p.circle(scatterDF["Longitude"], scatterDF["Latitude"],
         color=colors, fill_alpha=0.2, size=10)

output_file("Skeeterville.html", title="Skeeterville example")

show(p)


reset_output()
rm Skeeterville.html





########################################
# Nested bar chart
########################################

# https://bokeh.pydata.org/en/latest/docs/gallery/bar_pandas_groupby_nested.html
nestDF = train_complete[['Species','Trap','NumMosquitos']]
nestDF.head(n=15)

output_file("nested_bar.html")

nestDF.Species = nestDF.Species.astype(str)
group = nestDF.groupby(['Species', 'Trap'])

nestDF.groupby(['Species', 'Trap']).NumMosquitos


index_cmap = factor_cmap('spec_trp', palette=Spectral5, factors=sorted(nestDF.Species.unique()), end=1)

p = figure(plot_width=800, plot_height=300, title="Mosquito counts by Species and Trap",
           x_range=group, toolbar_location=None, tooltips=[("MPG", "@mosquito_count"), ("Spec,Trp", "@spec_trp")])

p.vbar(x='spec_trp', top='mosquito_count', width=1, source=group,
       line_color="white", fill_color=index_cmap)

p.y_range.start = 0
p.x_range.range_padding = 0.05
p.xgrid.grid_line_color = None
p.xaxis.axis_label = "Traps grouped by species"
p.xaxis.major_label_orientation = 1.2
p.outline_line_color = None

show(p)

reset_output()
rm nested_bar.html








########################################
# Weather line plot
########################################

output_file("line.html")
dates = np.array(train_complete['Date'], dtype=np.datetime64)
source = ColumnDataSource(data=dict(date=dates, temp=train_complete['Tmax_x']))

p = figure(plot_height=300, plot_width=800, tools="", toolbar_location=None,
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(dates[1500], dates[2500]))

p.line('date', 'temp', source=source)
p.yaxis.axis_label = 'Max Daily Temperature'

select = figure(title="Drag the middle and edges of the selection box to change the range above",
                plot_height=130, plot_width=800, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color="#efefef")

range_rool = RangeTool(x_range=p.x_range)
range_rool.overlay.fill_color = "navy"
range_rool.overlay.fill_alpha = 0.2

select.line('date', 'temp', source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_rool)
select.toolbar.active_multi = range_rool

show(column(p, select))


reset_output()
rm line.html



















########################################
# Distance from trap to spray site
########################################
train.head()
spray.head()
















########################################
# Modeling with scikit learn
########################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score



# Train / test split
X = train_complete.loc[:, train_complete.columns != 'WnvPresent']
y = train_complete['WnvPresent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Training evaluation metrics
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")



# Cross validation pedictions
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)





confusion_matrix(y_train_5, y_train_pred)
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)


















########################################
# Modeling with H2O
########################################
import h2o
h2o.init()












