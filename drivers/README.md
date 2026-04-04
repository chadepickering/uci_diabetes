# Drivers

This directory contains JDBC drivers required for R-Snowflake connectivity.
JAR files are excluded from version control due to file size.

## Setup

Download the Snowflake JDBC driver:
```bash
curl -o drivers/snowflake-jdbc.jar \
  https://repo1.maven.org/maven2/net/snowflake/snowflake-jdbc/3.15.0/snowflake-jdbc-3.15.0.jar
```

This JAR is required to run the Part 1 R/Quarto reports which connect 
to Snowflake via RJDBC. It is not required for Part 2 Python components 
which use snowflake-connector-python instead.