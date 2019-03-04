"""
Parse register file
"""

import pandas as pd
import scipy as sp

project_dir = '/faststorage/jail/project/'

def parse_reg_file(filename):
    df = pd.read_csv(filename, encoding = 'latin1' )
    
    # Set an index
    df["sampleID"] = df.sampleID.astype(str)
    df.set_index('sampleID')
    
    # Formal handling of types 
    
    
    # TODO: More types
    
    return df

def parse_fam_file(filename):
    df = pd.read_table(filename, delim_whitespace=True, header=None)

    # Set an index
    df[1] = df[1].astype(str)
    df.set_index(1)
    
    # Formal handling of types 
    
    # TODO: More types

    return df


def generate_fam_file(out_filename, phen_id, 
                      reg_filename=project_dir+'Register/current/NCRR_concat_genetics.csv', 
                      ref_fam_filename=project_dir+'DBS1to23/imatthei_imputation_merge/woBLACKLIST_newQC/phase3/cobg_dir_genome_wide/dbsALL-qc.hg19.ch.fl.bgs.fam', 
                      anc_filename=project_dir+'NCRR-PRS/ncrr_genetics/ancestry/ipsych_indiv_pred_anc.txt',
                      filter_EUR_ancestry=True, is_case_control=True, use_controls=True, w_covariates=True,
                      sample_frac=1):
    reg_df = parse_reg_file(reg_filename)
    ref_fam_df = parse_fam_file(ref_fam_filename)
    reg_df = reg_df.drop_duplicates("sampleID")
    ref_fam_df = ref_fam_df.drop_duplicates(1)
    
    #Filter for EUR ancestry
    if filter_EUR_ancestry:
        anc_df = pd.read_table(anc_filename, delim_whitespace=True)
        anc_df["IID"] = anc_df["IID"].astype(str)
        filt_anc_df = anc_df[anc_df['IS_EUR']==1]
        ref_fam_df = ref_fam_df[ref_fam_df[1].isin(filt_anc_df["IID"])]
        print ('Filtered for ancestry')
    
    #Choose a random sample
    if sample_frac<1:
        filt_reg_df = reg_df.sample(frac=sample_frac, replace=False)
    else:
        filt_reg_df = reg_df
        
    if is_case_control:
        if use_controls:
            filt_reg_df = filt_reg_df[sp.logical_or(filt_reg_df['kontrol2012I']==1, filt_reg_df[phen_id]==1)]
            
        else:
            filt_reg_df = filt_reg_df[~sp.isnan(filt_reg_df[phen_id])]


        #Filter data frames for values that we will use
        filt_fam_df = ref_fam_df[ref_fam_df[1].isin(filt_reg_df["sampleID"])]
        filt_reg_df = filt_reg_df[filt_reg_df["sampleID"].isin(filt_fam_df[1])]
        
        #Set everything to 0
        filt_fam_df[5].values[:] = 1
        
        #Seting cases to 1
        case_filt_reg_df = filt_reg_df[filt_reg_df[phen_id]==1]
        filt_fam_df[5].values[filt_fam_df[1].isin(case_filt_reg_df["sampleID"])]=2
        
        assert filt_reg_df['sampleID'].size==filt_fam_df[1].size,'Unequal sizes of register and family table (problems with parsing)'
        
        print ('Retained %d cases and %d controls'%(sp.sum(filt_fam_df[5]==2), sp.sum(filt_fam_df[5]==1)))
    else:
        raise NotImplementedError()
    
    with open(out_filename+".filter", 'w') as f:
        filt_fam_df.to_csv(f, ' ',header=False, index=False)

    with open(out_filename+".phen", 'w') as f:
        filt_fam_df[[0,1,5]].to_csv(f, ' ',header=['FID','IID','PHEN'], index=False)
    
        if w_covariates:
            #create a dataframe
            #Always include sex
            d = {'IID':filt_fam_df[1].values[:],'FID':filt_fam_df[0].values[:], 'SEX':filt_fam_df[4].values[:],}
            cov_df = pd.DataFrame(data=d)
            cov_df.to_csv(f, ' ',index=False)
                

    
        
    