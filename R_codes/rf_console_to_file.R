con <- file("test.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

# This will echo all input and not truncate 150+ character lines...
source("RF_Modeling_trial_test_loop.R", echo=TRUE, max.deparse.length=10000)

# Restore output to console
sink() 
sink(type="message")