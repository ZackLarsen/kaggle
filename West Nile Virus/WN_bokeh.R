
# Zack Larsen
# November 4, 2018

# West nile virus prediction Kaggle competition

# Target variable is presence of west nile virus
# Submission should have the following form:
# Id,WnvPresent
# 1,0
# 2,1
# 3,0.9
# 4,0.2

# Potential model attributes:
# Distance from spray locations that day
# Number of mosquitos of particular species present at trap previously


# Setup -------------------------------------------------------------------

library(pacman)
p_load(tidyverse, geosphere, ggplot2, ggvis, ggmap, rbokeh,
       dplyr, data.table, leaflet, microbenchmark, feather, e1071,
       psych)

proj <- '~/Desktop/Projects/Kaggle/West Nile/'
setwd(proj)

# Load data ---------------------------------------------------------------

# Samples
trainHead <- fread('train.csv',nrow=100)
testHead <- fread('test.csv',nrow=100)
weatherHead <- fread('weather.csv',nrow=100)
sprayHead <- fread('spray.csv',nrow=100)

# Real thing - using microbenchmark to time fread vs. read.csv:
# https://www.r-bloggers.com/timing-in-r/
write_feather(train, "train.feather") # Just doing this so we can benchmark it
microbenchmark(read.csv('test.csv'), times = 10, unit = "s")
microbenchmark(fread('test.csv'), times = 10, unit = "s")
read = microbenchmark(base = read.csv('test.csv'),
                      fread = fread('test.csv'),
                      feather_reader = read_feather("train.feather"),
                      times = 10, 
                      unit = "ms")
microbenchmark:::autoplot.microbenchmark(read)
microbenchmark:::boxplot.microbenchmark(read)

# Using feather as the serialization format - it is clearly 
# far faster than read_csv and even fread!
write_feather(train, "train.feather")
train_feather = read_feather("train.feather")
train_feather %>% 
  head()


train <- fread('train.csv')
test <- fread('test.csv')
weather <- fread('weather.csv')
spray <- fread('spray.csv')

train %>% 
  head()

test %>% 
  head()

spray %>% 
  head()

weather %>% 
  head()


# NA check and descriptives -----------------------------------------------

# Check for missing values
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))
sapply(spray, function(x) sum(is.na(x))) # Time has 584 NA values
spray %>% 
  filter(is.na(Time))
sapply(weather, function(x) sum(is.na(x)))

# Get number of rows with NA's in ANY column:
spray %>% 
  summarise(na_count = sum(is.na(.)))

map(train, ~sum(is.na(.)))

train %>%
  select(everything()) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.))))

# Row-wise
apply(train, MARGIN = 1, function(x) sum(is.na(x)))

# This is very slow
# train %>%
#   rowwise %>%
#   summarise(NA_per_row = sum(is.na(.)))


# Descriptive stats:
train %>% 
  head()

train %>% 
  glimpse()

dim(train)

sapply(train, class)

cbind(freq=table(train$Species), percentage=prop.table(table(train$Species))*100)

summary(train)

sapply(train[,10:11], sd)
sapply(train[,c(10,11)], sd) # Alternatively, using different slicing

apply(train[,10:11], 2, skewness)
apply(train[,10:11], 2, kurtosis)

cor(train[,10:12])

describe(train[,10:12]) # From psych package


# EDA ---------------------------------------------------------------------

train %>% 
  glimpse()

weather %>% 
  glimpse()


# Divide weather stations
station1 <- weather %>% 
  filter(Station == 1)

station2 <- weather %>% 
  filter(Station == 2)


# Analyze date ranges
min(train$Date) # "2007-05-29"
max(train$Date) # "2013-09-26"

min(spray$Date) # "2011-08-29"
max(spray$Date) # "2013-09-05"

min(weather$Date) # "2007-05-01"
max(weather$Date) # "2014-10-31"

min(test$Date) # "2008-06-11"
max(test$Date) # "2014-10-02"



# West Nile cases
train %>% 
  filter(WnvPresent == 1) %>% 
  group_by(Species) %>% 
  summarise(n())

train %>% 
  filter(WnvPresent == 1) %>% 
  select(Species, Trap, NumMosquitos) %>% 
  group_by(Species, Trap) %>% 
  summarise(skeeterCount = sum(NumMosquitos)) %>% 
  arrange(Species,desc(skeeterCount))

# Mosquitos per trap:
ggplot(data = train) +
  geom_boxplot(mapping = aes(x = reorder(Trap, NumMosquitos, FUN = median), y = NumMosquitos)) +
  coord_flip()


# How many days do we have train and weather data for?
length(intersect(train$Date, weather$Date)) # 95


# Narrow down training features to potentially relevant ones:
train %>% 
  select(Date, Trap, Latitude, Longitude, Species, AddressAccuracy, NumMosquitos, WnvPresent) %>% 
  head()


spray %>% 
  group_by(Date) %>% 
  distinct(Latitude, Longitude) %>% 
  summarise(locations = n()) # 10 unique spray dates, mostly in 2013




train %>% 
  add_count(Species) %>% 
  head()

train %>% 
  count(Species, Trap, sort = TRUE)


# Categorical heatmap
ggplot(train, aes(Trap, Date)) + 
  geom_tile(aes(fill = NumMosquitos),colour = "white")
#  + scale_fill_manual(values=c("red", "blue", "black"))


# Trap Map -----------------------------------------------------------------
# Make list of unique traps:
traps <- train %>% 
  select(Trap, Latitude, Longitude) %>% 
  unique() %>% 
  arrange()

traps %>% 
  head()

leaflet() %>%
  addTiles() %>%
  addMarkers(lat = traps$Latitude,
             lng = traps$Longitude,
             popup = 'Skeeter Trap')


# Make an icon to plot instead of the default pointer
# https://rstudio.github.io/leaflet/markers.html
SkeeterIcon <- makeIcon(iconUrl = '/Users/zacklarsen/Desktop/Learning/Projects/Kaggle/West Nile/skeeter.jpg',
                      iconWidth = 20, iconHeight = 20,
                      iconAnchorX = 20, iconAnchorY = 20)
SkeeterIcon

leaflet() %>%
  addTiles() %>%
  addMarkers(lat = traps$Latitude,
             lng = traps$Longitude,
             icon = SkeeterIcon)


# Mapping the total number of mosquitos per trap
skeeterCounts <- train %>% 
  group_by(Trap, Latitude, Longitude, AddressNumberAndStreet) %>%
  summarise(skeeterCount = sum(NumMosquitos))

sapply(skeeterCounts,mean)

gmap(lat = 41.84, lng = -87.69, zoom = 11, width = 700, height = 600) %>%
  ly_points(Longitude, Latitude, data = skeeterCounts, alpha = 0.8, col = "red",
            size = skeeterCount/500,
            hover = c(Trap, AddressNumberAndStreet, skeeterCount))



# Spray map ---------------------------------------------------------------
spray %>% 
  head()

sites <- spray %>% 
  distinct(Latitude, Longitude)

sites %>% 
  nrow()


# Preview the map here:
#gmap(lat = 41.857908, lng = -87.669147, zoom = 11, width = 700, height = 600)

# The real thing (with data plotted):
gmap(lat = 41.857908, lng = -87.669147, zoom = 11, width = 700, height = 600) %>%
  ly_points(Longitude, Latitude, data = spray, alpha = 0.1, col = "red",
            hover = Time) %>% 
  tool_box_select()





# Heatmap -----------------------------------------------------------------

figure() %>% 
  ly_hexbin(spray$Longitude, spray$Latitude)


# Heat map using leaflet (kind of)
leaflet(spray) %>% 
  addTiles() %>% 
  addMarkers(clusterOptions = markerClusterOptions())



# Weather plots -----------------------------------------------------------
weather %>% 
  select(Date, Station, Tmax, Tmin, Tavg) %>% 
  arrange(Station, Date) %>% 
  head(n=10)

min(weather$Date)
max(weather$Date)

# Make time series of weather data
tsDF <- station1 %>% select(Date, Tmax, Tmin, Tavg, DewPoint, WetBulb)
tsDF$Date <- as.Date(tsDF$Date)
tsDF %>% head(n=10)

library(xts)
dyXTS <- xts(x=tsDF$Tmax,order.by=tsDF$Date)
dyXTS

# Cool sliding window plot with time series data:
# https://www.htmlwidgets.org/showcase_dygraphs.html
library(dygraphs)
dygraph(dyXTS, main = "Chicago Temperatures") %>% 
  dyRangeSelector(dateWindow = c("2007-05-01", "2014-10-31"))





tsDF %>% head()

p <- figure(title = 'Chicago Weather') %>% 
  ly_lines(Date, Tmax, data = tsDF, hover = c(Date, Tavg, DewPoint, WetBulb)) %>%
  y_axis(label = 'Temperature')
p


p2 <- figure(title = 'Chicago Weather') %>% 
  ly_points(Date, DewPoint, data = tsDF, hover = c(Date, Tavg, DewPoint, WetBulb)) %>%
  y_axis(label = 'Dew Point')
p2








# Distance from trap to spray ---------------------------------------------
distm(c(-87.80099, 41.95469), c(-87.73978, 41.96052), fun = distHaversine)
distHaversine(c(-87.80099, 41.95469), c(-87.73978, 41.96052))

fivedist <- train %>% 
  select(Date, Trap, Latitude, Longitude) %>% 
  inner_join(spray, by="Date") %>% 
  select(Date, Trap, Latitude.x, Longitude.x, Latitude.y, Longitude.y) %>%
  rename("TrapLat"=Latitude.x, "TrapLon"=Longitude.x, "SprayLat"=Latitude.y, "SprayLon"=Longitude.y) %>% 
  head()

fivedist$distance<-distHaversine(fivedist[,3:4], fivedist[,5:6])
fivedist



Distances <- train %>% 
  select(Date, Trap, Latitude, Longitude) %>% 
  inner_join(spray, by="Date") %>% 
  select(Date, Trap, Latitude.x, Longitude.x, Latitude.y, Longitude.y) %>%
  rename("TrapLat"=Latitude.x, "TrapLon"=Longitude.x, "SprayLat"=Latitude.y, "SprayLon"=Longitude.y)

Distances$Haversine <- distHaversine(Distances[,3:4], Distances[,5:6])

Distances %>% 
  head(n=20)





# Link previous sprays to test data ---------------------------------------























# Test data transformations for predictions -------------------------------

# Below is our training format, so test should be the same but
# with no column for WnvPresent:
trainHead %>% 
  inner_join(station1, by ="Date") %>% 
  inner_join(station2, by = "Date") %>% 
  select(-one_of(c('Address','Block','Street','AddressNumberAndStreet'))) %>% 
  head()

testHead %>% 
  head()

testHead %>% 
  inner_join(traps, by="Trap") %>% 
  head()

testHead %>% 
  inner_join(station1, by ="Date") %>% 
  inner_join(station2, by = "Date") %>% 
  select(-one_of(c('Address','Block','Street','AddressNumberAndStreet'))) %>% 
  head()
  





# Scatterplot matrix ------------------------------------------------------

library("PerformanceAnalytics")
chart.Correlation(train[,10:12], histogram=TRUE, pch=19)



# Modeling ----------------------------------------------------------------

train %>% 
  head()
















