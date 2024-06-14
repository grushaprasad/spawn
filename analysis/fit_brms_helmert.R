library(ggplot2)
library(tidyverse)
library(brms)
library(lme4)
library(bayestestR)
set.seed(7)
library(argparse)

## Directory stuff
indir='/datalake/CLaP/SPAWN/CoNLL2024/preprocessed/'
outdir='/datalake/CLaP/SPAWN/CoNLL2024/brms_objects/'

## Argument parser

parser <- ArgumentParser()

parser$add_argument("--train", help = "train0.0, train0.1, train1.0, train10.0")
parser$add_argument("--reanalysis", help = "start, uncertainty1, uncertainty10")
parser$add_argument("--noise", help = "sduniform-0.2-0.5, sdnormal-0.35-1.0, sdnormal-0.35-2.0")
parser$add_argument("--giveup", help = "giveup100, giveup1000")

args <- parser$parse_args()

train <- args$train
reanalysis <- args$reanalysis
noise <- args$noise
giveup <- args$giveup

# train <- 'train0.0'
# reanalysis <- 'start'
# noise <- 'sduniform-0.2-0.5'
# giveup <- 'giveup100'
# m <- 'ep'

## Load in data
fname <- paste0(indir,paste(train,reanalysis,noise,giveup, sep='_'),'.csv')
dat <- read.csv(fname)

## Specify prior
prior <- c(prior("normal(-4.595,1.5)", class = Intercept),
           prior("normal(0,2)", class = b),  
           prior("normal(0,5)", class = sd))

## Fit brms models, get samples, get bayesfactor

models <- c('ep', 'wd', 'wd2')

for(m in models){
  curr_dat <- dat %>%
    filter(model==m,
           part_type == 'passive_part') %>%
    select(passive, c1, c2, c3, part_id, sent_id)
  
  ## Fit model
  curr_fit <-  brm(formula = passive ~ c1 + c2 + c3 + 
                     (1 + c1 + c2 + c3 | part_id) + 
                     (1 + c1 + c2 + c3 | sent_id),
                   data = curr_dat,
                   family = bernoulli(link = "logit"),   
                   prior = prior,
                   cores = 4,
                   iter = 40000,  ## need to compute BF; 10000 for quick
                   seed = 7,
                   control = list(max_treedepth = 20, adapt_delta = 0.95),
                   save_pars = save_pars(all = TRUE)
                   )
  
  fit_fname <- paste0(outdir,
                      'brm_',
                      paste(train,reanalysis,noise,giveup, sep='_'),
                      '.rds')
  
  saveRDS(curr_fit, fit_fname)
  
  ## Get Bayes factor
  
  bf_fname <- paste0(outdir,
                      'bf_',
                      paste(train,reanalysis,noise,giveup, sep='_'),
                      '.rds')
  
  curr_bf <- bayesfactor_parameters(curr_fit)
  saveRDS(curr_bf, bf_fname)
  
  ## Get posterior samples
  draws_fname <- paste0(outdir,
                        'draws_',
                        paste(train,reanalysis,noise,giveup, sep='_'),
                        '.rds')
  draws <- as_draws_df(curr_fit)
  
  saveRDS(draws, draws_fname)
  
  ## Summarize posterior
  posterior_summ <- draws %>%
    dplyr::select(b_Intercept, b_c1, b_c2, b_c3, `.chain`, `.draw`, `.iteration`) %>%
    mutate(amv = b_Intercept + b_c1*3/4,
           rrc = b_Intercept + b_c1*-1/4 + b_c2*2/3,
           prog_rrc = b_Intercept + b_c1*-1/4 + b_c2*-1/3 + b_c3*1/2,
           urc = b_Intercept + b_c1*-1/4 + b_c2*-1/3 + b_c3*-1/2) %>%
    dplyr::select(amv,rrc,prog_rrc, urc, `.chain`, `.draw`, `.iteration`) %>%
    gather(key='cond', value = 'logodds', amv,rrc,prog_rrc, urc) %>%
    mutate(prob = plogis(logodds))%>%
    group_by(cond) %>%
    summarise(mean_logodds = mean(logodds),
              lower_logodds = quantile(logodds, 0.025)[[1]],
              upper_logodds = quantile(logodds, 0.975)[[1]],
              mean_prob = mean(prob),
              lower_prob = quantile(prob, 0.025)[[1]],
              upper_prob = quantile(prob, 0.975)[[1]]) %>%
    mutate(cond = toupper(cond),
           cond = ifelse(cond == 'PROG_RRC', 'ProgRRC', 
                         ifelse(cond == 'URC', 'FRC', cond)),
           cond = factor(cond, levels = c('RRC', 'ProgRRC', 'FRC', 'AMV')),
           model = m,
           train = train,
           reanalysis = reanalysis,
           sd = noise,
           giveup= giveup)
  
  posterior_summ_fname <- paste0(outdir,
                                 'postsumm_',
                                 paste(train,reanalysis,noise,giveup, sep='_'),
                                 '.rds')
  saveRDS(posterior_summ, posterior_summ_fname)
  
  #Clear memory before next iteration
  rm(draws) 
  rm(curr_fit)
}


