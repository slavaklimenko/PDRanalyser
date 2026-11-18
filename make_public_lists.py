#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_public_lists.py

Adds groups of published systems to the PDR database and writes a public database with only these systems.

Groups (systems in the order of the paper tables; names written as in the database):
    MW_CO_Klimenko2024   Milky Way CO sightlines, Klimenko et al. 2024, MNRAS 533, 1367, Table 1   (28)
    MW_CI_Klimenko2024   Milky Way C I sightlines (Jenkins & Tripp 2011 sample),
                         Klimenko et al. 2024, MNRAS 533, 1367, Table A1                          (56)
    QSO_Klimenko2024     CO-bearing DLAs, Klimenko et al. 2024, MNRAS 533, 1367, Table 1          (7)

Usage (in the PDRanalyser folder):
    python make_public_lists.py [database] [public file]
        database     default PDR_database_full.fits; the groups are added and the file is saved again
        public file  default PDR_database_public.fits; only the systems of the groups above,
                     without the internal 'comment' notes
"""

import sys

from PDR_database import PDRdatabase

PUBLIC_GROUPS = {
    'MW_CO_Klimenko2024': [
        'HD24534',     'HD27778',     'HD147888',    'HD185418',
        'HD206267',    'HD207198',    'HD210121',    'HD210839',
        'HD12323',     'HD15137',     'HD23478',     'HD24190',
        'HD24398',     'HD30122',     'HD96675',     'HD99872',
        'HD137595',    'HD144965',    'HD147683',    'HD148937',
        'HD190918',    'HD192035',    'HD198781',    'HD203532',
        'HD220057',    'HD62542',     'HD28975',     'HD29647',
    ],
    'MW_CI_Klimenko2024': [
        'CPD-592603',  'HD15137',     'HD23478',     'HD24190',
        'HD24534',     'HD37903',     'HD40893',     'HD69106',
        'HD75309',     'HD88115',     'HD91824',     'HD91983',
        'HD93205',     'HD93222',     'HD93843',     'HD94454',
        'HD99857',     'HD99872',     'HD102065',    'HD103779',
        'HD104705',    'HD108002',    'HD108639',    'HD114886',
        'HD115071',    'HD115455',    'HD116852',    'HD121968',
        'HD122879',    'HD124314',    'HD144965',    'HD147683',
        'HD147888',    'HD152590',    'HD157857',    'HD177989',
        'HD185418',    'HD192639',    'HD195965',    'HD198781',
        'HD201345',    'HD202347',    'HD203374',    'HD203532',
        'HD206267',    'HD206773',    'HD207198',    'HD208440',
        'HD209339',    'HD210809',    'HD210839',    'HD218915',
        'HD219188',    'HD220057',    'HD224151',    'HDE303308',
    ],
    'QSO_Klimenko2024': [
        'J0000+0048',  'J0857+1855',  'J1047+2057',  'J1237+0647',
        'J1439+1118',  'J1705+3543',  'J2331-0908',
    ],
}


def add_public_groups(db, replace=True):
    """add the groups of PUBLIC_GROUPS to db (replacing groups with the same name)"""
    missing = {g: [s for s in ids if s not in set(db.systems['sys_id'])] for g, ids in PUBLIC_GROUPS.items()}
    missing = {g: v for g, v in missing.items() if v}
    if missing:
        raise KeyError(f'systems not in the database: {missing}')
    for g, ids in PUBLIC_GROUPS.items():
        if replace:
            db.remove_group(g)
        db.add_group(g, ids)
    return db


if __name__ == '__main__':
    dbfile = sys.argv[1] if len(sys.argv) > 1 else 'PDR_database_full.fits'
    pubfile = sys.argv[2] if len(sys.argv) > 2 else 'PDR_database_public.fits'

    db = PDRdatabase.read(dbfile)
    add_public_groups(db)
    db.write(dbfile)
    print(f'{dbfile}: added groups {list(PUBLIC_GROUPS)}')

    pub = db.subset(list(PUBLIC_GROUPS))
    pub.systems['comment'] = ''          # internal notes are not published
    pub.write(pubfile)
    print(f'{pubfile}: {pub}')
