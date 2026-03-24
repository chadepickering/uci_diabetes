# connections.R
# Shared Snowflake connection and data load for Part 1 Quarto reports
# Sourced at the top of each .qmd file via source(here::here("reports/part1/R/connections.R"))

options(java.parameters = "--add-opens=java.base/java.nio=ALL-UNNAMED")
library(rJava)
library(RJDBC)
library(tidyverse)
library(here)

# Snowflake JDBC connection
drv <- JDBC(
  driverClass = "net.snowflake.client.jdbc.SnowflakeDriver",
  classPath = "/Users/cepickering/Documents/Large Folders/Git/uci_diabetes/drivers/snowflake-jdbc.jar"
)

con <- dbConnect(
  drv,
  url = paste0(
    "jdbc:snowflake://ar29154.us-central1.gcp.snowflakecomputing.com/",
    "?db=UCI_DIABETES&schema=MARTS&warehouse=COMPUTE_WH&role=TRANSFORMER"
  ),
  user = Sys.getenv("SNOWFLAKE_USER"),
  password = Sys.getenv("SNOWFLAKE_PASSWORD")
)

# Load marts table into R session
df <- dbReadTable(con, "DIABETES_FEATURES") |>
  janitor::clean_names()

# Close connection after loading
dbDisconnect(con)