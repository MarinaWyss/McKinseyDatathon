library(tidyverse)
library(caret)

set.seed(123)

buildingOwnership <- read.csv("train_Vh587l8/building_ownership.csv")
buildingStructure <- read.csv("train_Vh587l8/building_structure.csv")
demographics <- read.csv("train_Vh587l8/ward_demographic_data.csv")
train <- read.csv("train_Vh587l8/train.csv")


## CLASS IMBALANCE ##
train$damage_grade <- as.factor(train$damage_grade)

trainBalanced <- downSample(x = train,
                   y = train$damage_grade)

trainBalanced$Class <- NULL


## COMBINE DATA ##
fullTrainData <- left_join(trainBalanced, buildingOwnership)
fullTrainData <- left_join(fullTrainData, buildingStructure)
fullTrainData <- left_join(fullTrainData, demographics)

# just a couple of missings - drop
fullTrainData <- fullTrainData[complete.cases(fullTrainData), ]

# ordinal to numeric
fullTrainData <- fullTrainData %>% 
  mutate(land_surface_condition = case_when(land_surface_condition == "Flat" ~ 1, 
                                            land_surface_condition == "Moderate slope" ~ 2,
                                            land_surface_condition == "Steep slope" ~ 3),
         income_range_in_thousands = case_when(income_range_in_thousands == "0-10" ~ 1,
                                               income_range_in_thousands == "10-20" ~ 2, 
                                               income_range_in_thousands == "20-30" ~ 3,
                                               income_range_in_thousands == "30-50" ~ 4,
                                               income_range_in_thousands == "50+" ~ 5),
         damage_grade = as.numeric(damage_grade))


# standardize
toStandardize <- fullTrainData %>% 
  dplyr::select(count_families, count_floors_pre_eq, age_building, plinth_area_sq_ft, 
         height_ft_pre_eq, household_count, income_range_in_thousands, avg_hh_size)

notStandardized <- fullTrainData %>% 
  dplyr::select(-count_families, -count_floors_pre_eq, -age_building, -plinth_area_sq_ft, 
                -height_ft_pre_eq, -household_count, -income_range_in_thousands, -avg_hh_size)

library(robustHD)
standardized <- standardize(toStandardize, centerFun = mean, scaleFun = sd)

fullStandardized <- cbind(notStandardized, standardized)


## FEATURE SELECTION ##

# polr model
library(MASS)

fullStandardized$damage_grade <- as.factor(fullStandardized$damage_grade)

model1 <- polr(damage_grade ~ . -building_id -ward_id -district_id -vdcmun_id, 
               data = fullStandardized)

summary_table <- coef(summary(model1))
pval <- pnorm(abs(summary_table[, "t value"]), lower.tail = FALSE) * 2
summary_table <- cbind(summary_table, "p value" = round(pval, 3))
summary_table


# corrplot
library(corrplot)
corrData <- fullTrainDataNumeric %>% 
  as.matrix() %>% 
  cor()

corrplot(corrData,
         tl.col = "black", 
         order = "hclust",
         tl.cex = 0.8,
         tl.srt = 70)


## FIRST MODELS ##

# random forest
library(ranger)

n_features <- length(setdiff(names(fullStandardized), "damage_grade"))

# train a default random forest model
rf1 <- ranger(
  damage_grade ~ ., 
  data = fullStandardized,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# tuning
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .63, .8),                       
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = damage_grade ~ ., 
    data            = fullStandardized, 
    num.trees       = n_features * 10,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)


# tuned model
rf2 <- ranger(
  damage_grade ~ ., 
  data = fullStandardized,
  mtry = 12,
  min.node.size = 5,
  replace = TRUE,
  sample.fraction = 0.5,
  respect.unordered.factors = "order",
  seed = 123
)

## VISUALIZATIONS ##
library(vip)

viPlot <- vip(rf2, num_features = 25, bar = FALSE)


## TEST ON TEST DATA ##
test <- read.csv("test_yeQNDNV/test.csv")

fullTestData <- left_join(test, buildingOwnership)
fullTestData <- left_join(fullTestData, buildingStructure)
fullTestData <- left_join(fullTestData, demographics)

# just a couple of missings - drop
fullTestData <- fullTestData[complete.cases(fullTestData), ]

# ordinal to numeric
fullTestData <- fullTestData %>% 
  mutate(land_surface_condition = case_when(land_surface_condition == "Flat" ~ 1, 
                                            land_surface_condition == "Moderate slope" ~ 2,
                                            land_surface_condition == "Steep slope" ~ 3),
         income_range_in_thousands = case_when(income_range_in_thousands == "0-10" ~ 1,
                                               income_range_in_thousands == "10-20" ~ 2, 
                                               income_range_in_thousands == "20-30" ~ 3,
                                               income_range_in_thousands == "30-50" ~ 4,
                                               income_range_in_thousands == "50+" ~ 5))


# standardize
toStandardizeTest <- fullTestData %>% 
  dplyr::select(count_families, count_floors_pre_eq, age_building, plinth_area_sq_ft, 
                height_ft_pre_eq, household_count, income_range_in_thousands, avg_hh_size)

notStandardizedTest <- fullTestData %>% 
  dplyr::select(-count_families, -count_floors_pre_eq, -age_building, -plinth_area_sq_ft, 
                -height_ft_pre_eq, -household_count, -income_range_in_thousands, -avg_hh_size)

standardizedTest <- standardize(toStandardizeTest, centerFun = mean, scaleFun = sd)

fullTestStandardized <- cbind(notStandardizedTest, standardizedTest)


# prediction
predictions <- predict(rf2, data = fullTestStandardized)$predictions

rounded 

submit <- fullTestStandardized %>% 
  select(building_id)

submit <- cbind(submit, rounded)

write.csv(submit, "solutionFinal.csv")


## LoCATION ##

# where are these districts?

regionData <- read.csv("District and VDC codes of Nepal.csv")

regionData <- regionData %>% 
  rename(district_id = District_code)

geoData <- left_join(fullTrainData, regionData)


geoDataPlot1 <- geoData %>% 
  dplyr::select(damage_grade, Zone) %>%
  group_by(Zone) %>% 
  summarise(mean = mean(damage_grade)) %>% 
  mutate_if(is.numeric, ~ round(., digit = 2))
  

plot1 <- ggplot(data = geoDataPlot1, 
                aes(x = reorder(Zone, mean), y = mean, fill = mean)) +
  geom_bar(stat = "identity", color = "black") + 
  scale_fill_gradient(low = "blue", high = "red") +
  geom_text(aes(label = mean, vjust = 3)) +
  xlab("Zone") +
  ylab("Mean damage grade") +
  ggtitle("Mean earthquake damage grade by zone")

plot1


geoDataPlot2 <- geoData %>% 
  dplyr::select(damage_grade, Geographical.Region) %>%
  group_by(Geographical.Region) %>% 
  summarise(mean = mean(damage_grade)) %>% 
  mutate_if(is.numeric, ~ round(., digit = 2))


plot2 <- ggplot(data = geoDataPlot2, 
                aes(x = reorder(Geographical.Region, mean), y = mean, fill = mean)) +
  geom_bar(stat = "identity", color = "black")  + 
  scale_fill_gradient(low = "blue", high = "red") +
  geom_text(aes(label = mean, vjust = 3)) +
  xlab("Geography type") +
  ylab("Mean damage grade") +
  ggtitle("Mean earthquake damage grade by geography type")

plot2

