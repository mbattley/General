#!/usr/local/python/bin/python
"""
Take a list of RA Dec and look for them in the NGTS database
"""
import argparse as ap
from collections import OrderedDict
from astropy.coordinates import SkyCoord
import astropy.units as u
import pymysql

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

if __name__ == "__main__":
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
    with pymysql.connect(host='ngtsdb', db='ngts_archive') as cur:
        for obj_id in objects:
            qry_args = objects[obj_id]
            cur.execute(qry, qry_args)
            results = cur.fetchall()
            print(obj_id, objects[obj_id])
            for row in results:
                print(row)
                try:
                    prod_ID = row[0][50:57]
                    cand_ID = row[0][58:64]
                    print('prod_ID = {}'.format(prod_ID))
                    print('cand_ID = {}'.format(cand_ID))
                except:
                    print('Skipped {}'.format(row[0]))
            print('\n')