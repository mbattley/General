#!/usr/local/python/bin/python
"""
Take a list of RA Dec and look for them in the NGTS database
"""
import argparse as ap
from collections import OrderedDict
from astropy.coordinates import SkyCoord
import astropy.units as u
import pymysql
import csv
import time
from astropy.io import ascii
import multiprocessing as multip
from astropy.table import Table

# pylint: disable=invalid-name

def argParse():
    """
    Parse command line args
    """
    p = ap.ArgumentParser()
    p.add_argument('infile',
                   help='name of file containing obj_id ra dec')
    p.add_argument('source',
                   help='source table to crossmatch',
                   choices=['NGTS', 'NASA'])
    return p.parse_args()

def getObjectsFromFile(infile):
    """
    Read in the targets from file
    """
    objects = OrderedDict()
    f = open(infile).readlines()
    for line in f:
        if not line.startswith('#'):
            sp = line.split()
            assert len(sp) == 3, "File format must be obj_id ra dec"
            obj_id = sp[0]
            # check for hms or deg coords
            try:
                ra = float(sp[1])
                dec = float(sp[2])
            except ValueError:
                coords = " ".join(sp[1:])
                s = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.degree))
                ra = s.ra.deg
                dec = s.dec.deg
            objects[obj_id] = (ra, dec)
    return objects

def get_observation_IDS(obj_id,objects,qry):
    with pymysql.connect(host='ngtsdb', db='ngts_archive') as cur:
        exists = 0
        qry_args = objects[obj_id]
        cur.execute(qry, qry_args)
        results = cur.fetchall()
        print(obj_id, objects[obj_id])
#            for row in results:
#                print(row)
        try:
#                print(results[0])
            prod_ID = results[0][0][50:57]
            cand_ID = results[0][0][58:64]
#                print('prod_ID = {}'.format(prod_ID))
#                print('cand_ID = {}'.format(cand_ID))
            exists = 1
        except:
#                print('Skipped {}'.format(obj_id))
            prod_ID = '0'
            cand_ID = '0'
        with open('NGTS_Young_Star_xmatch_10k_final.csv','a') as f:
            w = csv.writer(f)
            w.writerow([obj_id,prod_ID,cand_ID])
#            sensitivity_table.add_row([obj_id,prod_ID,cand_ID])
        print('\n')
        current_time = time.time()
        print('Elapsed time = {}s'.format(current_time - start))
        return exists

if __name__ == "__main__":
    start = time.time()
    args = argParse()
    objects = getObjectsFromFile(args.infile)
    if args.source == 'NGTS':
        qry = """
            SELECT
            CONCAT("https://ngts.warwick.ac.uk/monitor/opis_view_cand/", r.prod_id, "/", c.obj_id) as url
            FROM catalogue AS c
            LEFT JOIN orion_runs AS r ON c.cat_prod_id=r.cat_prod_id
            WHERE GREATCIRCLE(c.ra_deg, c.dec_deg, %s, %s) < 10.0/3600.0
            """
    else:
        qry = """
            SELECT *
            FROM opis_known_planets
            WHERE GREATCIRCLE(ra_deg, dec_deg, %s, %s) < 5.0/3600.0
            """
    opis_links = []
    no_targets = 0
    print('Total number of processors on your machine is: {}'.format(multip.cpu_count()))
    with open('NGTS_Young_Star_xmatch_10k_final.csv','w') as f:
        w = csv.writer(f)
        w.writerow(['DR2_ID','Prod_ID','Cand_ID'])
#    sensitivity_table = Table({'DR2_ID':[],'Prod_ID':[],'Cand_ID':[]},names=['DR2_ID','Prod_ID','Cand_ID'])
#    sensitivity_table['DR2_ID'] = sensitivity_table['DR2_ID'].astype(str)
#    sensitivity_table['Prod_ID'] = sensitivity_table['Prod_ID'].astype(str)
#    sensitivity_table['Cand_ID'] = sensitivity_table['Cand_ID'].astype(str)
    poolv = multip.Pool(multip.cpu_count()) 
    results = [poolv.apply_async(get_observation_IDS, args=(obj_id,objects,qry)) for obj_id in objects]
#    with pymysql.connect(host='ngtsdb', db='ngts_archive') as cur:
#        for obj_id in objects:
#            qry_args = objects[obj_id]
#            cur.execute(qry, qry_args)
#            results = cur.fetchall()
#            print(obj_id, objects[obj_id])
##            for row in results:
##                print(row)
#            try:
##                print(results[0])
#                prod_ID = results[0][0][50:57]
#                cand_ID = results[0][0][58:64]
##                print('prod_ID = {}'.format(prod_ID))
##                print('cand_ID = {}'.format(cand_ID))
#            except:
##                print('Skipped {}'.format(obj_id))
#                prod_ID = '0'
#                cand_ID = '0'
#            with open('NGTS_Young_Star_xmatch.csv','a') as f:
#                w = csv.writer(f)
#                w.writerow([obj_id,prod_ID,cand_ID])
##            sensitivity_table.add_row([obj_id,prod_ID,cand_ID])
#            no_targets += 1
#            print('Number of targets analysed = {}'.format(no_targets))
#            print('\n')
#            current_time = time.time()
#            print('Elapsed time = {}s'.format(current_time - start))
#    ascii.write(sensitivity_table,'NGTS_Young_Star_xmatch_from_table.csv', format='csv', overwrite = True)
     
    poolv.close()
    output = [p.get() for p in results]
    print(output)
    finish = time.time()
    print('Elapsed time = {}s'.format(finish - start))