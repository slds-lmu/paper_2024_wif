library(mgcv)
library(fairmodels)
library(xtable)
library(fairadapt)
set.seed(940984)
source("code/func_help.R")

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 0. Preparation of data
# 1. Learn warping models
# 2. Train models in both worlds
# 3. Apply warping on test data
# 4. fairadapt
# 5. Predict in warped world
# 6. Evaluation
#----------------------------------#

save_plots <- TRUE
height <- 5

#----------------------------------#
#### 0. Preparation of data ####
#----------------------------------#

# Load preprocessed COMPAS data by Plecko et al.: https://github.com/dplecko/fairadapt/blob/main/data/compas.rda
load(file="data/compas.rda")
compas <- compas[, c("sex", "age", "priors_count", "c_charge_degree", "race", "two_year_recid")]
compas$c_charge_degree <- as.numeric(compas$c_charge_degree == "M")
cols <- c("sex", "age", "priors_count", "c_charge_degree")

# Train-Test split
train_ids <- sample(seq_len(nrow(compas)), size = 0.8*nrow(compas), replace = FALSE)
dat_train <- compas[train_ids,] 
dat_test <- compas[-train_ids,]

# dim(dat_train)
# dim(dat_test)

disadv_class <- "Non-White"
adv_class <- "White"
adv_rows <- dat_train$race != disadv_class
disadv_rows <- dat_train$race == disadv_class

# Sub-data sets for each race
compas_adv <- dat_train[adv_rows,]
compas_disadv <- dat_train[disadv_rows,]

#----------------------------------#
#### 1. Learn warping models and warp training data ####
#----------------------------------#

### Priors
# advantaged and disadvantaged model for warping
c_mod_priors_disadv <- glm(priors_count ~ 1 + age + sex, family = poisson(link="log"), data = compas_disadv)
c_mod_priors_adv <- glm(priors_count ~ 1 + age + sex, family = poisson(link="log"), data = compas_adv)
# summary(c_mod_priors_disadv)
# summary(c_mod_priors_adv)

# Warp disadvantaged amount values to advantaged model
c_priors_star_disadv <- warp_model(dat_x = compas_disadv, 
                          model_x = c_mod_priors_disadv, 
                          model_y = c_mod_priors_adv)

### Degree
c_mod_degree_disadv <- glm(c_charge_degree ~ 1 + age + sex, family = binomial, data = compas_disadv)
c_mod_degree_adv <- glm(c_charge_degree ~ 1 + age + sex, family = binomial, data = compas_adv)
# summary(c_mod_degree_disadv)
# summary(c_mod_degree_adv)

c_degree_star_disadv <- warp_model(dat_x = compas_disadv, 
                           model_x = c_mod_degree_disadv, 
                           model_y = c_mod_degree_adv)

### Risk
c_mod_recid_disadv <- glm(two_year_recid ~ 1 + age + sex + c_charge_degree + priors_count, 
                    family = binomial, data = compas_disadv)
c_mod_recid_adv <- glm(two_year_recid ~ 1 + age + sex + c_charge_degree + priors_count, 
                    family = binomial, data = compas_adv)
# summary(c_mod_recid_disadv)
# summary(c_mod_recid_adv)

compas_disadv_warped <- compas_disadv
compas_disadv_warped[,"c_charge_degree"] <- c_degree_star_disadv
compas_disadv_warped[,"priors_count"] <- c_priors_star_disadv

c_recid_star_disadv <- warp_model(dat_x = compas_disadv_warped, 
                            model_x = c_mod_recid_disadv, 
                            model_y = c_mod_recid_adv)

# compas_disadv_warped[,"Risk"] <- factor(round(c_recid_star_disadv), levels = c(0,1),
#                                    labels=c("bad", "good"))

compas_disadv_warped[,"two_year_recid"] <- as.numeric(round_risk(c_recid_star_disadv, compas_adv$two_year_recid))-1
#levels(compas_disadv_warped[,"two_year_recid"]) <- c(0,1)

# all.equal(compas_disadv_warped[,"Risk_old"], compas_disadv_warped[,"Risk"])
compas_warped <- rbind(compas_adv, compas_disadv_warped)
compas_warped <- compas_warped[order(as.numeric(rownames(compas_warped))),]
head(compas_warped, 20)
head(dat_train[order(as.numeric(rownames(dat_train))),], 20)

summary(compas_disadv_warped)
summary(compas_disadv)
summary(compas_adv)

summary(dat_train)
summary(compas_warped)

#----------------------------------#
#### 2. Train models in both worlds ####
#----------------------------------#

g_mod_real <- glm(two_year_recid ~ ., family = binomial, data = dat_train)
g_mod_real_wo_pa <- glm(two_year_recid ~ . - race, family = binomial, data = dat_train)
g_mod_warped <- glm(two_year_recid ~ ., family = binomial, data = compas_warped)
g_mod_warped_wo_pa <- glm(two_year_recid ~ . - race, family = binomial, data = compas_warped)

summary(g_mod_real)
summary(g_mod_real_wo_pa)
summary(g_mod_warped)
summary(g_mod_warped_wo_pa)


#----------------------------------#
# 3. Warp test data with training data warping model ####
#----------------------------------#

dat_test_real_disadv <- dat_test_warped_disadv <- dat_test[dat_test$race==disadv_class,]
dat_test_real_adv <- dat_test[dat_test$race==adv_class,]

# Warp disadvantaged "charge degree" values to advantaged model
dat_test_warped_disadv[,"c_charge_degree"] <- warp_new_data(dat_new = dat_test_warped_disadv, 
                                                  model_x = c_mod_degree_disadv, 
                                                  model_y = c_mod_degree_adv,
                                                  target = "c_charge_degree")

dat_test_warped_disadv[,"c_charge_degree"] <- round(dat_test_warped_disadv[,"c_charge_degree"])

# Warp disadvantaged "priors count" values to advantaged model
dat_test_warped_disadv[,"priors_count"] <- warp_new_data(dat_new = dat_test_warped_disadv, 
                                                  model_x = c_mod_priors_disadv, 
                                                  model_y = c_mod_priors_adv,
                                                  target = "priors_count")

dat_test_warped_disadv[,"priors_count"] <- round(dat_test_warped_disadv[,"priors_count"])


# Warp disadvantaged "Risk" values to advantaged model
# str(dat_test_warped_disadv[,"Risk"])
#dat_test_warped_disadv[,"two_year_recid"] <- dat_test_warped_disadv[,"two_year_recid"]
# str(dat_test_warped_disadv[,"Risk"])
dat_test_warped_disadv[,"two_year_recid"] <- warp_new_data(dat_new = dat_test_warped_disadv, 
                                                 model_x = c_mod_recid_disadv, 
                                                 model_y = c_mod_recid_adv,
                                                 target = "two_year_recid")

#----------------------------------#
#### 4. fairadapt ####
#----------------------------------#


adj.mat <- c(
  0, 0, 1, 1, 0, 1, 0, # sex
  0, 0, 1, 1, 0, 1, 0, # age
  0, 0, 0, 0, 0, 1, 0, # priors
  0, 0, 0, 0, 0, 1, 0, # c_charge
  0, 0, 1, 1, 0, 1, 0, # race
  0, 0, 0, 0, 0, 0, 1, # y
  0, 0, 0, 0, 0, 0, 0  # pseudo-target Z
)

vars <- c(colnames(dat_train), "Z")
adj.mat <- matrix(adj.mat, 
                  nrow = length(vars), 
                  ncol = length(vars),
                  dimnames = list(vars, vars), 
                  byrow = TRUE)

dat_train$Z <- rnorm(nrow(dat_train))
dat_test$Z <- rnorm(nrow(dat_test))

dat_train$race <- relevel(dat_train$race, ref = "White")
dat_test$race <- relevel(dat_test$race, ref = "White")

mod <- fairadapt(Z ~ ., # Z
                 train.data = dat_train,
                 test.data = dat_test, 
                 prot.attr = "race", 
                 adj.mat = adj.mat,
                 visualize.graph=TRUE#, 
                 #res.vars = "hours_per_week"
)

adapt.train <- adaptedData(mod)
adapt.test  <- adaptedData(mod, train = FALSE)
summary(mod)
summary(adapt.train)
summary(adapt.test)

# adapt_train <- adapt.train
adapt_train <- adapt.train[,vars[1:6]]
adapt_train$race <- dat_train$race
#adapt_test <- adapt.test
adapt_test <- adapt.test[,vars[1:6]]
adapt_test$race <- dat_test$race
summary(adapt_train)
summary(adapt_test)


g_mod_adapt <- glm(two_year_recid ~., family = binomial, 
                   data = adapt_train)
summary(g_mod_adapt)

#----------------------------------#
#### 5. Predict in warped world ####
#----------------------------------#

# Predict target for test data
pred_disadv_warped <- predict(g_mod_warped, newdata = dat_test_warped_disadv, type="response")
pred_adv_warped <- predict(g_mod_warped, newdata = dat_test_real_adv, type="response")

pred_disadv_real <- predict(g_mod_real, newdata = dat_test_real_disadv, type="response")
pred_adv_real <- predict(g_mod_real, newdata = dat_test_real_adv, type="response")

pred_disadv_adapt <- predict(g_mod_adapt, newdata = adapt_test[adapt_test$race=="Non-White",], type="response")
pred_adv_adapt <- predict(g_mod_adapt, newdata = adapt_test[adapt_test$race=="White",], type="response")

#----------------------------------#
#### 6. Evaluation ####
#----------------------------------#

#--------------#
# 1) Test performance

mean(round(pred_disadv_real) == dat_test_real_disadv$two_year_recid)
mean(round(pred_adv_real) == dat_test_real_adv$two_year_recid)

mean(round(pred_disadv_warped) == round(dat_test_warped_disadv$two_year_recid))
mean(round(pred_adv_warped) == dat_test_real_adv$two_year_recid)


#--------------#
# 2) Mapping
#   a) Features: Which features vary most between the 2 worlds?

# MSEs of normalized values
mse_vec <- mse_func_col(dat_test_warped_disadv, dat_test_real_disadv, cols=cols[2:4])
print(mse_vec)

#----------#
#   b) Observations: Which observations change the most between the 2 worlds?

dat_eval_map <- dat_test_real_disadv[,cols]
dat_eval_map$p_warped <- dat_test_warped_disadv[,"priors_count"]
dat_eval_map$d_warped <- dat_test_warped_disadv[,"c_charge_degree"]
dat_eval_map$mse_row_wr <- mse_func_row(dat_test_warped_disadv, dat_test_real_disadv, cols=cols[2:4])
dat_eval_map[,-1] <- round(dat_eval_map[,-1], 3)
head(dat_eval_map[order(dat_eval_map$mse, decreasing=TRUE),])
tail(dat_eval_map[order(dat_eval_map$mse, decreasing=TRUE),])

mod_warp_change_sim <- gam(mse_row_wr~s(age) + s(priors_count) + c_charge_degree, dat=dat_eval_map)
summary(mod_warp_change_sim)
plot(mod_warp_change_sim, pages=1)

#------------#
# 3) ML model
#   b) Compare predictions in the 2 worlds => similar to 2b)

dat_eval_pred <- dat_test_real_disadv[,cols]
dat_eval_pred$p_warped <- dat_test_warped_disadv[,"priors_count"]
dat_eval_pred$d_warped <- dat_test_warped_disadv[,"c_charge_degree"]
dat_eval_pred$pf_real <- round(pred_disadv_real,4)
dat_eval_pred$pf_warped <- round(pred_disadv_warped,4)
dat_eval_pred$diff_f_wr <- round(pred_disadv_warped-pred_disadv_real,4)

head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),])
head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=FALSE),])

# Same for advantaged individuals
dat_eval_pred_adv <- dat_test_real_adv[,cols]
dat_eval_pred_adv$pm_real <- round(pred_adv_real,4)
dat_eval_pred_adv$pm_warped <- round(pred_adv_warped,4)
dat_eval_pred_adv$diff_m_wr <- round(pred_adv_warped-pred_adv_real,4)
head(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, decreasing=TRUE),])
head(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, decreasing=FALSE),])

## ## ## ##
# fairadapt
dat_eval_pred_adapt <- dat_test_real_disadv[,cols]
dat_eval_pred_adapt$p_warped <- adapt_test[adapt_test$race=="Non-White","priors_count"]
dat_eval_pred_adapt$c_warped <- adapt_test[adapt_test$race=="Non-White","c_charge_degree"]
dat_eval_pred_adapt$pf_real <- round(pred_disadv_real,2)
dat_eval_pred_adapt$pf_warped <- round(pred_disadv_adapt,2)
dat_eval_pred_adapt$diff_f_wr <- round(pred_disadv_adapt-pred_disadv_real,2)

head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing=TRUE),])
head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing=FALSE),])

# Same for male individuals
dat_eval_pred_adv_adapt <- dat_test_real_adv[,cols]
dat_eval_pred_adv_adapt$pm_real <- round(pred_adv_real,2)
dat_eval_pred_adv_adapt$pm_warped <- round(pred_adv_adapt,2)
dat_eval_pred_adv_adapt$diff_m_wr <- round(pred_adv_adapt-pred_adv_real,2)
head(dat_eval_pred_adv_adapt[order(dat_eval_pred_adv_adapt$diff_m_wr, decreasing=TRUE),])
head(dat_eval_pred_adv_adapt[order(dat_eval_pred_adv_adapt$diff_m_wr, decreasing=FALSE),])

## ## ## ##
if(save_plots){
  pdf("plots/compas_2b.pdf", width=height, height=height)
}
boxplot(dat_eval_pred$diff_f_wr, dat_eval_pred_adv$diff_m_wr, xlab=NULL, 
        main="Prediction difference warped-real", ylab="Prediction difference")
axis(1,at=c(1:2) ,labels=c(disadv_class, adv_class), las=1)
dev.off()
mean(dat_eval_pred$diff_f_wr)
mean(dat_eval_pred_adv$diff_m_wr)

t.test(dat_eval_pred$diff_f_wr)$p.value
t.test(dat_eval_pred_adv$diff_m_wr)$p.value

## ## ## ##
# fairadapt
if(save_plots){
  pdf("plots/compas_2b_adapt.pdf", width=height, height=height)
}
boxplot(dat_eval_pred_adapt$diff_f_wr, dat_eval_pred_adv_adapt$diff_m_wr, xlab=NULL, 
        main="Prediction difference adapt-real", ylab="Prediction difference")
#,ylim=c(-0.29,0.25)
axis(1,at=c(1:2) ,labels=c(disadv_class, adv_class), las=1)
dev.off()

mean(dat_eval_pred_adapt$diff_f_wr)
mean(dat_eval_pred_adv_adapt$diff_m_wr)

t.test(dat_eval_pred_adapt$diff_f_wr)$p.value
t.test(dat_eval_pred_adv_adapt$diff_m_wr)$p.value

# ## ## ## ##
# # Regression of change on features - disadvantaged
# mod_pred_change_disadv <- gam(diff_f_wr~s(age) + s(priors_count) + c_charge_degree, dat=dat_eval_pred)
# summary(mod_pred_change_disadv)
# if(save_plots){
#   pdf("plots/compas_1.pdf")
# }
# plot(mod_pred_change_disadv, pages=1, main = "Partial effect on prediction difference")
# dev.off()
# 
# # Regression of change on features - advantaged
# mod_pred_change_adv <- gam(diff_m_wr~s(age) + s(priors_count) + c_charge_degree, dat=dat_eval_pred_adv)
# summary(mod_pred_change_adv)
# if(save_plots){
#   pdf("plots/compas_3.pdf")
# }
# plot(mod_pred_change_adv, pages=1, main="Partial effect")
# dev.off()

## ## ## ##
# fairadapt

## ## ## ##
# Compare ranks: advantaged
pred_order_adv_real <- as.numeric(rownames(dat_eval_pred_adv[order(dat_eval_pred_adv$pm_real, decreasing=TRUE),]))
advantaged_df_ranks_real <- data.frame(ID = pred_order_adv_real, rank_real = seq_len(length(pred_order_adv_real)))

pred_order_adv_warped <- as.numeric(rownames(dat_eval_pred_adv[order(dat_eval_pred_adv$pm_warped, decreasing=TRUE),]))
advantaged_df_ranks_warped <- data.frame(ID = pred_order_adv_warped, rank_warped = seq_len(length(pred_order_adv_warped)))

advantaged_df_ranks <- merge(advantaged_df_ranks_warped, advantaged_df_ranks_real, by = "ID")
# barplot(table(advantaged_df_ranks$rank_real-advantaged_df_ranks$rank_warped))
boxplot(dat_eval_pred_adv$pm_real, dat_eval_pred_adv$pm_warped, xlab=NULL, 
        main=paste0("Risk predictions",adv_class))
axis(1,at=c(1:2) ,labels=c("advantaged real", "advantaged warped"), las=1)
for (i in seq_len(nrow(advantaged_df_ranks))){
  segments(1, dat_eval_pred_adv$pm_real[i], 2, dat_eval_pred_adv$pm_warped[i], col = "gray", lty = "solid")
}

# disadvantaged
pred_order_disadv_real <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_real, decreasing=TRUE),]))
disadvantaged_df_ranks_real <- data.frame(ID = pred_order_disadv_real, rank_real = seq_len(length(pred_order_disadv_real)))

pred_order_disadv_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_warped, decreasing=TRUE),]))
disadvantaged_df_ranks_warped <- data.frame(ID = pred_order_disadv_warped, rank_warped = seq_len(length(pred_order_disadv_warped)))

disadvantaged_df_ranks <- merge(disadvantaged_df_ranks_warped, disadvantaged_df_ranks_real, by = "ID")
# barplot(table(disadvantaged_df_ranks$rank_real-disadvantaged_df_ranks$rank_warped))

if(save_plots){
  pdf("plots/compas_4b.pdf", width=height, height=height)
}
boxplot(dat_eval_pred$pf_real, dat_eval_pred$pf_warped, xlab=NULL, 
        main=paste0("Risk predictions ",disadv_class), ylab="Risk prediction")
axis(1,at=c(1:2) ,labels=c("Real world", "Warped world"), las=1)
for (i in seq_len(nrow(dat_eval_pred))){
  segments(1, dat_eval_pred$pf_real[i], 2, dat_eval_pred$pf_warped[i], col = "gray", lty = "solid", lwd=0.5)
}
for (i in seq_len(nrow(dat_eval_pred))){
  if(i %% 50 == 0) segments(1, dat_eval_pred$pf_real[i], 2, dat_eval_pred$pf_warped[i], col = "black", lty = "solid")
}
dev.off()

## ## ## ##
# fairadapt

# Compare ranks: adv
pred_order_adv_adapt <- as.numeric(rownames(dat_eval_pred_adv_adapt[order(dat_eval_pred_adv_adapt$pm_warped, decreasing=TRUE),]))
adv_df_ranks_adapt <- data.frame(ID = pred_order_adv_adapt, 
                                  rank_warped = seq_len(length(pred_order_adv_adapt)))

adv_df_ranks_adapt <- merge(adv_df_ranks_adapt, advantaged_df_ranks_real, by = "ID")
# barplot(table(adv_df_ranks_adapt$rank_real-adv_df_ranks_adapt$rank_warped))
boxplot(dat_eval_pred_adv_adapt$pm_real, dat_eval_pred_adv_adapt$pm_warped, xlab=NULL, main="Risk predictions male")
axis(1,at=c(1:2) ,labels=c("male real", "male adapt"), las=1)
for (i in seq_len(nrow(adv_df_ranks_adapt))){
  segments(1, dat_eval_pred_adv_adapt$pm_real[i], 2, dat_eval_pred_adv_adapt$pm_warped[i], col = "gray", lty = "solid")
}

# disadv
pred_order_disadv_adapt <- as.numeric(rownames(dat_eval_pred_adapt[order(dat_eval_pred_adapt$pf_warped, decreasing=TRUE),]))
disadv_df_ranks_adapt <- data.frame(ID = pred_order_disadv_adapt, 
                                    rank_warped = seq_len(length(pred_order_disadv_adapt)))

disadv_df_ranks_adapt <- merge(disadv_df_ranks_adapt, disadvantaged_df_ranks_real, by = "ID")
# barplot(table(disadv_df_ranks_adapt$rank_real-disadv_df_ranks$rank_warped))
if(save_plots){
  pdf("plots/compas_4b_adapt.pdf", width=height, height=height)
}
boxplot(dat_eval_pred_adapt$pf_real, dat_eval_pred_adapt$pf_warped, xlab=NULL, 
        main =paste0("Risk predictions ",disadv_class), ylab="Risk prediction"#,
        #ylim=c(0.35, 0.9))
)
axis(1,at=c(1:2) ,labels=c("Real world", "Adapt world"), las=1)
for (i in seq_len(nrow(dat_eval_pred_adapt))){
  segments(1, dat_eval_pred_adapt$pf_real[i], 2, dat_eval_pred_adapt$pf_warped[i], col = "gray", lty = "solid")
}
for (i in seq_len(nrow(dat_eval_pred_adapt))){
  # segments(1, dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing = TRUE),]$pf_real[j],
  #          2, dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing = TRUE),]$pf_warped[j],
  #          col = "black", lty = "solid")
  if(i %% 50 == 0) segments(1, dat_eval_pred_adapt$pf_real[i], 2, dat_eval_pred_adapt$pf_warped[i], col = "black", lty = "solid")
}

dev.off()
## ## ## ##

plot(density(compas$age[compas$race=="White"]), lwd=2)
lines(density(compas$age[compas$race=="Non-White"]), col="blue", lty=2, lwd=2)
legend("topright", legend=c("non-white", "white"), col = c("blue", "black"), lty = c(1:2), lwd=2)

#----------------------------------#
#### Tables for Paper ####
#----------------------------------#

head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=FALSE),])
tail(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=FALSE),])

head(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, decreasing=FALSE),])
tail(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, decreasing=FALSE),])

summary(g_mod_real)
summary(g_mod_warped)
summary(g_mod_adapt)

# RPID 
table1 <- xtable(head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, 
                                decreasing=FALSE),c(1,2,3,4,5,6,7,8,9)]),
       digits=c(0,0,0,0,0,0,0,2,2,2))
table2 <- xtable(tail(dat_eval_pred[order(dat_eval_pred$diff_f_wr, 
                                decreasing=FALSE),c(1,2,3,4,5,6,7,8,9)]),
       digits=c(0,0,0,0,0,0,0,2,2,2))

table3 <- xtable(head(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, 
                                    decreasing=FALSE),c(1,2,3,4,5,6,7)]),
       digits=c(0,0,0,0,0,2,2,2))
table4 <- xtable(tail(dat_eval_pred_adv[order(dat_eval_pred_adv$diff_m_wr, 
                                    decreasing=FALSE),c(1,2,3,4,5,6,7)]),
       digits=c(0,0,0,0,0,2,2,2))


# Fairadapt 
tablef1 <- xtable(head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, 
                                      decreasing=FALSE),]),
       digits=c(0,0,0,0,0,0,0,2,2,2))
tablef2 <- xtable(tail(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, 
                                      decreasing=FALSE),]),
       digits=c(0,0,0,0,0,0,0,2,2,2))


tablef3 <- xtable(head(dat_eval_pred_adv_adapt[order(dat_eval_pred_adv_adapt$diff_m_wr, 
                                           decreasing=FALSE),]),
       digits=c(0,0,0,0,0,2,2,2))
tablef4 <- xtable(tail(dat_eval_pred_adv_adapt[order(dat_eval_pred_adv_adapt$diff_m_wr, 
                                          decreasing=FALSE),]),
       digits=c(0,0,0,0,0,2,2,2),
       include.rownames=FALSE)

# Print without row names
print(table1, include.rownames=FALSE)
print(table2, include.rownames=FALSE)
print(table3, include.rownames=FALSE)
print(table4, include.rownames=FALSE)

# 
# # Print without row names
# print(tablef1, include.rownames=FALSE)
# print(tablef2, include.rownames=FALSE)
# print(tablef3, include.rownames=FALSE)
# print(tablef4, include.rownames=FALSE)


