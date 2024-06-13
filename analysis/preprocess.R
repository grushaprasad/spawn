library(tidyverse)

## Directory specfication
setwd('/home/gprasad/SPAWN/spawn/')
outdir='/datalake/CLaP/SPAWN/CoNLL2024/preprocessed/'  #where preprocessed files will be stored
indir = '../predictions/'    #where predictions are stored

## Load in stimulus files
lists <- c('list_1A', 'list_1B', 'list_1C', 'list_1D',
           'list_1A_rev', 'list_1B_rev', 'list_1C_rev', 'list_1D_rev',
           'list_2A', 'list_2B', 'list_2C', 'list_2D',
           'list_2A_rev', 'list_2B_rev', 'list_2C_rev', 'list_2D_rev',
           'list_3A', 'list_3B', 'list_3C', 'list_3D',
           'list_3A_rev', 'list_3B_rev', 'list_3C_rev', 'list_3D_rev',
           'list_4A', 'list_4B', 'list_4C', 'list_4D',
           'list_4A_rev', 'list_4B_rev', 'list_4C_rev', 'list_4D_rev')


cond_dat_list <- list()

i <- 1
for(fname in lists){
  curr_fname <- paste('./data/stimuli/', fname, '.tsv', sep='')
  curr_dat <- read.csv(curr_fname, sep='\t') %>%
    mutate(sent_id = c(0:(n()-1)),
           list = fname) 
  
  cond_dat_list[[i]] <- curr_dat
  i <- i+1
}

cond_dat <-dplyr::bind_rows(cond_dat_list)

## Load in model predictions

train <- c('train0.0', 'train0.1', 'train1.0', 'train10.0')
sds <- c('sduniform-0.2-0.5', 'sdnormal-0.35-1.0', 'sdnormal-0.35-2.0')
reanalysis <- c('start', 'uncertainty1', 'uncertainty10')
giveup <- c('giveup100', 'giveup1000')
models <- c('ep', 'wd', 'wd2')

combined_summ_list = list()
i <- 1 #index into combined_summ_list

for(r in reanalysis){
  for(t in train){
    for(s in sds){
      for(g in giveup){
        j<-1 #index into combined
        combined = list()
        for(m in models){
          fname=paste0(indir,paste(m,t,r,s,g,sep='_'), '.csv')
          if(file.exists(fname)){
            curr_dat <- read.csv(fname)
            curr_passive_parts <- unique(subset(curr_dat, passive == 1)$part_id)
            curr_dat <- curr_dat %>%
              mutate(list = str_replace(list, './data/stimuli/', ''),
                     list = str_replace(list, '.txt', '')) %>%
              merge(cond_dat, by = c('list', 'sent_id')) %>%
              mutate(c1 = ifelse(prime_type == 'amv', 3/4, -1/4),
                     c2 = ifelse(prime_type == 'amv', 0,
                                 ifelse(prime_type == 'rrc', 2/3, -1/3)),
                     c3 = ifelse(prime_type %in% c('amv', 'rrc'), 0,
                                 ifelse(prime_type == 'prog_rrc', 1/2, -1/2)),
                     part_type = ifelse(part_id %in% curr_passive_parts, 'passive_part', 'no_passive_part'),
                     train = t,
                     model = m,
                     reanalysis = r,
                     sd = s,
                     giveup = g)
            
            combined[[j]] <- curr_dat
            j<-j+1
          }
        }
        combined <- dplyr::bind_rows(combined)
        
        if (nrow(combined)>0){ ## model files existed
          combined_fname <- paste0(outdir,paste(t,r,s,g,sep='_'), '.csv')
          write.csv(combined, combined_fname)
          
          combined_summ <- combined %>%
            filter(part_type == 'passive_part') %>%
            group_by(model, prime_type) %>%
            mutate(prime_type = factor(prime_type,
                                       levels = c('amv', 'urc', 'prog_rrc', 'rrc'))) %>%
            summarise(prop_passive = mean(passive, na.rm = TRUE),
                      se_passive = sd(passive, na.rm=TRUE)/sqrt(n()),
                      n = n(),
                      .groups='drop') %>%
            mutate(train = t,
                   reanalysis = r,
                   sd = s,
                   giveup = g)
          
          combined_summ_list[[i]] <- combined_summ
          i <- i+1
        }
        
        
          
      }
    }
  }
}


all_combined_summ <- dplyr::bind_rows(combined_summ_list)
write.csv(all_combined_summ, paste0(outdir, 'raw_combined_summ', '.csv'))
write.csv(all_combined_summ, './analysis/raw_combined_summ.csv')







