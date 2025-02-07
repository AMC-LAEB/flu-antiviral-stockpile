# load libraries 
library(tidyverse)
library(tidybayes)
library(brms)
library(broom)
library(modelr)
library(broom.mixed)

# set working directory 
setwd("~/Dropbox/flu-antiviral-stockpile/")

# load data 
input_data <- read_csv("./manuscript/bayesian_analysis_input.csv")
input_data <- input_data %>% 
  mutate(bxm_resist_h1 = ifelse(bxm_resistance_label == 'h1_resist', 1, 0), 
         bxm_resist_h3 = ifelse(bxm_resistance_label == 'h3_resist', 1, 0)) %>% 
  mutate(treat_bxm6599 = ifelse(treatment == 'BXM65-99', 1, 0),
         treat_bxm0524 = ifelse(treatment == 'BXM05-24', 1, 0), 
         treat_bxm2564 = ifelse(treatment == 'BXM25-64', 1, 0)) %>%
  mutate(across(starts_with("perc_averted"), list("z" = ~scale(.)))) %>%
  mutate(across(starts_with("f_treatment"), list("z" = ~scale(.))))
glimpse(input_data)

# make random draws reproducible
set.seed(42)
# use the cmdstanr backend 
options(mc.cores = 4,  # use 4 cores
        brms.backend = "cmdstanr")
bayes_seed <- 42

# m1: country and pandemic random effects
# assuming no systematic variation (i.e. a certain country has a "biased" observed measurement under a certain pandemic)
m1_av_bf <- brmsformula(
  f_treatment_norm_z ~ 1 + avs_week + PEP + tat_delay + treat_window_extended + bxm_resist_h1 + bxm_resist_h3 + treat_bxm6599 + treat_bxm0524 + treat_bxm2564 + (1 + avs_week + PEP + tat_delay + treat_window_extended + bxm_resist_h1 + bxm_resist_h3 + treat_bxm6599 + treat_bxm0524 + treat_bxm2564|country) + (1 + avs_week + PEP + tat_delay + treat_window_extended + bxm_resist_h1 + bxm_resist_h3 + treat_bxm6599 + treat_bxm0524 + treat_bxm2564|pandemic),
  family = gaussian()
)

# get_prior(m1_bf, data = input_data)
prior1 <- c(
  set_prior("normal(0,1)", class = 'Intercept'),
  set_prior("student_t(3,0,2.5)", class = 'b'),
  set_prior("normal(0,1)", class = 'sd', lb=0), # half-normal 
  set_prior("lkj(2)", class = 'cor'),
  set_prior("normal(0,1)", class = 'sigma', lb = 0) # half-normal 
)

m1_av <- brm(
  formula = m1_av_bf, 
  data = input_data, 
  prior = prior1,
  iter = 2000, 
  warmup = 1000, 
  chains = 4,
  cores = 4, 
  file = "./manuscript/m1_av_extz_rerun27jan.rds",
  silent = 1,
  backend = "cmdstanr", 
  threads = threading(2)
)

summary(m1_av)

# get coefficients
fe <- m1_av %>% 
  spread_draws(b_Intercept, sigma, b_avs_week, b_PEP, b_tat_delay, b_treat_window_extended, b_bxm_resist_h1, b_bxm_resist_h3, b_treat_bxm0524, b_treat_bxm2564, b_treat_bxm6599)
write.csv(fe, file = "./manuscript/av_m1_fixed_effects.csv")

re_pandemic <- m1_av %>% 
  spread_draws(r_pandemic[condition,term])
write.csv(re_pandemic, file = "./manuscript/av_m1_re_pandemic.csv")

re_country <- m1_av %>% 
  spread_draws(r_country[condition,term])
write.csv(re_country, file = "./manuscript/av_m1_re_country.csv")
