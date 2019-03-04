"""
Various tiny scripts to run on the cluster.
"""
import pandas as pd


def reparse_phen_file():
    pf = "test_phen.txt"
    phen_map = {}
    num_phens_found = 0
    with open(pf, 'r') as f:
        for line in f:
            l = line.split()
            fid = l[0]
            iid = l[1]
            sex = int(l[4])
            phen = int(l[5])
            if sex != 0 and phen != -9:
                phen_map[iid] = {'phen': phen, 'sex': sex, 'fid':fid}
                num_phens_found +=1
    print 'Found %d phenotypes'%num_phens_found
    
    
    def val_map(d):
        if d==2:
            return 1
        if d==1: 
            return 0
        else:
            return d
    
    of = "new_phen"
    with open(of, 'w') as f:
        f.write("FID IID PHEN\n")
        for iid in phen_map.keys():
            f.write("%s %s %d\n"%(phen_map[iid]['fid'], iid, val_map(phen_map[iid]['phen'])))
        
        
def 


def get_plink_iid_filter_file(filter_file, fam_file):
    filter_t = pd.read_table(filter_file, delim_whitespace=True)