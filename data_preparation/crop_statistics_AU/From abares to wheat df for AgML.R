library(tidyverse)

df = read.csv("fdp-beta-regional-historical.csv") #file origin https://www.agriculture.gov.au/sites/default/files/documents/fdp-beta-regional-historical.csv
head(df)

wheat = df %>% filter(Variable == 'Wheat produced (t)' | Variable == 'Wheat area sown (ha)')
wheat = wheat %>% select(-RSE)
wheat_df = wheat %>% pivot_wider(names_from = Variable , values_from = Value)
head(wheat_df)

wheat_df$Yield = wheat_df$`Wheat produced (t)`/wheat_df$`Wheat area sown (ha)` 
wheat_df = drop_na(wheat_df) %>% select(Year,ABARES.region,Yield)

head(wheat_df)

write.csv(wheat_df,'wheat_Australia.csv')

ggplot(wheat_df)+
  geom_line(aes(x=Year,y=Avg_Yield_tperha))+
  facet_wrap(~ABARESregion)+
  theme_bw()

