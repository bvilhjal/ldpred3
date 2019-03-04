"""
iPSYCH PRS code
"""
import sys
import os
sys.path.append(os.getenv("HOME")+"/REPOS/ldpred")
import random

from ldpred import *

import scipy as sp
import h5py

from plinkio import plinkfile
from itertools import izip


# Step 0. Define data files:
# Raw phenotype file

# Raw genotype file

# Raw summary stats file

# Pre-process files..           


# Step 1. Run LDpred out of the box

# Step 2. Adapt LDpred to data...



def main():
    
    
    
    
    pass


if __name__ == '__main__':
    main()
    
    


    
def calc_relatedness(plink_genot_file, hdf5_output_file, min_maf=0.05, sample_indiv_fraction=0.05, sample_snp_fraction=0.05):
    """
    Calculates:
        - The relatedness matrix for the given genotype dataset.

    :param plink_genot_file: Genotype file in plink format
    :param hdf5_output_file: 
    :param min_maf: 
    :param sample_indiv_fraction: 
    :param sample_snp_fraction: 
    :return: 
    
    """
    
    plinkf = plinkfile.PlinkFile(plink_genot_file)
    samples = plinkf.get_samples()
    
    print 'Calculating Principal components for genotype file %s' % plink_genot_file
    # Load genotypes
    print 'Loading genotypes'
    
    iids = [sample.iid for sample in samples]
    num_indivs = len(iids)
    
    print 'Found %d individuals'%num_indivs 
    
    indiv_filter = sp.random.random(num_indivs)<sample_indiv_fraction
           
    
    num_nt_issues = 0
    num_snps_used = 0
    print 'Iterating over BED file to calculate PC projections.'
    locus_list = plinkf.get_loci()
    
    
    norm_snps = []
    
    #Iterating over all SNPs
    for locus, row in izip( locus_list, plinkf):
        if random.random()<sample_snp_fraction:
            try:
                #Check rs-ID
    #             sid = '%d_%d'%(locus.chromosome,locus.bp_position)
                sid = locus.name
    
            except Exception: #Move on if rsID not found.
                continue
                    
            
            #Parse SNP
            snp = sp.array(row[indiv_filter], dtype='int8')
            
            # ... and fill in the blanks if necessary.
            bin_counts = row.allele_counts()
            if bin_counts[-1]>0:
                mode_v = sp.argmax(bin_counts[:2])
                snp[snp==3] = mode_v
            
            mean_g = sp.mean(snp)
            af = mean_g/2.0
            if af<min_maf or (1-af)<min:
                continue
    
            sd_g = sp.sqrt(af * (1 - af))
            # "Normalizing" the SNPs with the given allele frequencies
            norm_snp = (snp - mean_g) / sd_g
            norm_snp.shape = (num_indivs, )
    
            norm_snps.append(norm_snp)
    plinkf.close()
        
    norm_snps = sp.matrix(sp.concatenate(norm_snps,axis=1)) 
    
    h5f = h5py.open(hdf5_output_file,'w')
    h5f.
        
    print '%d SNPs were excluded from the analysis due to nucleotide issues.' % (num_nt_issues)
    print '%d SNPs were used for the analysis.' % (num_snps_used)
    
    return {'pcs': pcs, 'num_snps_used': num_snps_used, 'num_indivs':num_indivs, 'iids':iids}

