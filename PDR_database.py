#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDR_database.py

Database of observed absorption systems stored in a FITS file and handled as pandas DataFrames.
The public file PDR_database_public.fits contains the published systems.

The FITS file has four binary-table extensions, read into four DataFrames:

    SYSTEMS     one row per system
                sys_id, name, z_em, z_dla, telescope, year, ref, coord, progID, SIMBAD, SDSS,
                comment, full, mangastatus, Me_ind, mags (JSON), extra (JSON, rarely used attributes)
                ref and progID are lists written as 'a; b; c'; coord is JSON
    COMPONENTS  one row per velocity component
                sys_id, comp (0, 1, ...), z, z_err, z_err_minus (NaN if the error is symmetric)
    VALUES      one row per quantity (column density, T01, n, UV, Me, ...)
                sys_id, comp (-1 = whole system, 0, 1, ... = component), kind ('e' or 'physcond'),
                name, val, plus, minus,
                type   : 'm' measurement, 'u' upper limit, 'l' lower limit, 'f' fixed / no error
                format : 'log' or 'dec'
                b_val, b_plus, b_minus, b_type : Doppler parameter (NaN / '' if not set)
                f, ref : optional keywords of PDR_tools.e ('' if not set)
    GROUPS      membership of systems in groups (QSO, MW, LMC, ...)
                group, sys_id, comps ('all', '' for none, or a list like '0,2')

Values are stored once per system; groups only point to systems (and components), so editing a
value changes it in every group that contains the system.

Usage
-----
    from PDR_database import PDRdatabase

    db = PDRdatabase.read('PDR_database_public.fits')
    db.groups_list()                          # names of the groups
    db.table('QSO_Klimenko2024', ['H2', 'CO', 'CI'])        # pandas view: one row per system
    db.table('QSO_Klimenko2024', ['H2', 'T01'], comp=True)  # one row per component

    QSO = db.load('QSO_Klimenko2024')         # PDR_systems.sample object, used by PDR_exc

    # edit
    db.set_value('J0000+0048', 'H2', 20.45, 0.03)                     # system value
    db.set_value('J0000+0048', 'H2j0', 20.30, 0.02, comp=4)           # component value
    db.set_value('J0000+0048', 'CO', 13.1, type='u')                  # upper limit
    db.add_system('J1234+5678', z_em=2.9, z_dla=2.41, ref=['Me2026'])
    c = db.add_component('J1234+5678', z=2.4105)
    db.set_value('J1234+5678', 'H2', 19.2, 0.1, comp=c)
    db.add_group('my_sample', ['J0000+0048', 'J1234+5678'])
    db.add_to_group('my_sample', 'B0528-2505', comps=[0])
    db.remove_component('J1234+5678', c)
    db.write('my_database.fits')

    # the DataFrames can also be edited directly: db.systems, db.components, db.values, db.groups

Command line
------------
    python PDR_database.py export   [file]   build the FITS file from the private PDR_summary.py (default PDR_database_full.fits)
    python PDR_database.py selftest [file]   check that the FITS file reproduces PDR_summary.py exactly (needs PDR_summary.py)
"""

import contextlib
import inspect
import io
import json
import sys

import numpy as np
import pandas as pd

from PDR_tools import a, e

EXTENSIONS = ('SYSTEMS', 'COMPONENTS', 'VALUES', 'GROUPS')

SYSTEM_COLUMNS = ['sys_id', 'name', 'z_em', 'z_dla', 'telescope', 'year', 'ref', 'coord', 'progID',
                  'SIMBAD', 'SDSS', 'comment', 'full', 'mangastatus', 'Me_ind', 'mags', 'extra']
COMPONENT_COLUMNS = ['sys_id', 'comp', 'z', 'z_err', 'z_err_minus']
VALUE_COLUMNS = ['sys_id', 'comp', 'kind', 'name', 'val', 'plus', 'minus', 'type', 'format',
                 'b_val', 'b_plus', 'b_minus', 'b_type', 'f', 'ref']
GROUP_COLUMNS = ['group', 'sys_id', 'comps']

_STR_COLUMNS = {'sys_id', 'name', 'telescope', 'year', 'ref', 'coord', 'progID', 'SIMBAD', 'SDSS', 'comment',
                'full', 'mangastatus', 'Me_ind', 'mags', 'extra', 'kind', 'type', 'format', 'b_type', 'f',
                'group', 'comps'}
_INT_COLUMNS = {'comp'}
_LIST_SEP = '; '
# attributes of PDR_summary.qso stored in their own columns (everything else plain goes to 'extra')
_QSO_STANDARD = {'name', 'z_em', 'z_dla', 'telescope', 'year', 'ref', 'coord', 'progID', 'SIMBAD', 'SDSS',
                 'comment', 'full', 'mangastatus', 'Me_ind', 'm', 'e', 'comp'}


def _summary():
    """the private catalogue PDR_summary.py (only needed for 'export' and 'selftest')"""
    try:
        import PDR_summary
    except ImportError:
        raise ImportError('PDR_summary.py (the full catalogue of systems) is not available; '
                          'it is only needed to build or check the database. Use PDRdatabase.read() instead.') from None
    return PDR_summary


def _systems():
    import PDR_systems
    return PDR_systems


# =====================================================================================
# conversion: PDR_summary objects  <->  rows
# =====================================================================================

def _value_row(sys_id, comp, kind, el):
    c = el.col
    row = dict(sys_id=sys_id, comp=comp, kind=kind, name=el.name, val=float(c.val), plus=float(c.plus),
               minus=float(c.minus), type=c.type, format=c.repr,
               b_val=np.nan, b_plus=np.nan, b_minus=np.nan, b_type='', f='', ref='')
    if el.b is not None:
        row.update(b_val=float(el.b.val), b_plus=float(el.b.plus), b_minus=float(el.b.minus), b_type=el.b.type)
    if hasattr(el, 'f'):
        row['f'] = str(el.f)
    if hasattr(el, 'ref'):
        row['ref'] = str(el.ref)
    return row


def _make_e(row):
    """PDR_tools.e object from a VALUES row"""
    col = a(float(row['val']), float(row['plus']), float(row['minus']))
    col.type = row['type']
    col.repr = col.default_format = row['format']
    kw = {}
    if row['f']:
        kw['f'] = row['f']
    if row['ref']:
        kw['ref'] = row['ref']
    if isinstance(row['b_type'], str) and row['b_type']:
        kw['b'] = (float(row['b_val']), float(row['b_plus']), float(row['b_minus']))
    el = e(row['name'], col, **kw)
    if 'b' in kw:
        el.b.type = row['b_type']
    return el


def _comp_signature(c):
    def es(d):
        return tuple((k, v.col.val, v.col.plus, v.col.minus, v.col.type, v.col.repr,
                      None if v.b is None else (v.b.val, v.b.plus, v.b.minus, v.b.type)) for k, v in d.items())
    return (c.z, repr(c.z_err), es(c.e), es(c.physcond))


def _system_signature(q):
    plain = tuple(sorted((k, repr(v)) for k, v in vars(q).items() if k not in ('e', 'comp') and not isinstance(v, e)))
    vals = tuple((k, v.col.val, v.col.plus, v.col.minus, v.col.type, v.col.repr,
                  None if v.b is None else (v.b.val, v.b.plus, v.b.minus, v.b.type)) for k, v in q.e.items())
    return (plain, vals)


def _split(s):
    return [x for x in s.split(_LIST_SEP)] if s else []


class PDRdatabase:
    """database of absorption systems as four pandas DataFrames (see module docstring)"""

    def __init__(self, systems=None, components=None, values=None, groups=None):
        self.systems = systems if systems is not None else pd.DataFrame(columns=SYSTEM_COLUMNS)
        self.components = components if components is not None else pd.DataFrame(columns=COMPONENT_COLUMNS)
        self.values = values if values is not None else pd.DataFrame(columns=VALUE_COLUMNS)
        self.groups = groups if groups is not None else pd.DataFrame(columns=GROUP_COLUMNS)

    # ---------------------------------------------------------------- building from PDR_summary
    @classmethod
    def from_samples(cls, samples):
        """
        build the database from PDR_summary samples
            - samples : dict {group name: PDR_summary.sample}; groups are processed in this order.
                        Systems with the same name and the same data are stored once; if the data differ,
                        the later one gets sys_id = '<name>@<group>'.
        """
        sys_rows, comp_rows, val_rows, grp_rows = [], [], [], []
        versions = {}      # name -> list of (signature, sys_id)
        full = {}          # sys_id -> list of comp signatures (the stored components)
        for g, smp in samples.items():
            for q in smp.values():
                sig = _system_signature(q)
                known = versions.setdefault(q.name, [])
                sys_id = next((sid for s, sid in known if s == sig), None)
                if sys_id is None:
                    sys_id = q.name if not known else f'{q.name}@{g}'
                    known.append((sig, sys_id))
                    extra = {k: v for k, v in vars(q).items() if k not in _QSO_STANDARD and not isinstance(v, e)}
                    ref = q.ref if isinstance(q.ref, (list, tuple)) else ([q.ref] if q.ref else [])
                    sys_rows.append(dict(sys_id=sys_id, name=q.name, z_em=float(q.z_em), z_dla=float(q.z_dla),
                                         telescope=str(q.telescope), year=str(q.year),
                                         ref=_LIST_SEP.join(map(str, ref)), coord=json.dumps(q.coord) if q.coord else '',
                                         progID=_LIST_SEP.join(map(str, q.progID)), SIMBAD=str(q.SIMBAD),
                                         SDSS=str(q.SDSS), comment=str(q.comment), full=str(q.full),
                                         mangastatus=str(q.mangastatus), Me_ind=str(q.Me_ind),
                                         mags=json.dumps(q.m) if q.m else '',
                                         extra=json.dumps(extra) if extra else ''))
                    for el in q.e.values():
                        val_rows.append(_value_row(sys_id, -1, 'e', el))
                    full[sys_id] = []
                # components: add the ones not stored yet, then record which ones this group uses
                comps = []
                for c in q.comp:
                    cs = _comp_signature(c)
                    used = [i for i in comps]
                    idx = next((i for i, s in enumerate(full[sys_id]) if s == cs and i not in used), None)
                    if idx is None:
                        idx = len(full[sys_id])
                        full[sys_id].append(cs)
                        zerr, zerrm = (c.z_err if isinstance(c.z_err, tuple) else (c.z_err, np.nan))
                        comp_rows.append(dict(sys_id=sys_id, comp=idx, z=float(c.z), z_err=float(zerr),
                                              z_err_minus=float(zerrm)))
                        for el in c.e.values():
                            val_rows.append(_value_row(sys_id, idx, 'e', el))
                        for el in c.physcond.values():
                            val_rows.append(_value_row(sys_id, idx, 'physcond', el))
                    comps.append(idx)
                grp_rows.append(dict(group=g, sys_id=sys_id, comps=comps))
        # 'all' where a group uses every stored component in the stored order
        for r in grp_rows:
            r['comps'] = 'all' if r['comps'] == list(range(len(full[r['sys_id']]))) else ','.join(map(str, r['comps']))
        db = cls(pd.DataFrame(sys_rows, columns=SYSTEM_COLUMNS), pd.DataFrame(comp_rows, columns=COMPONENT_COLUMNS),
                 pd.DataFrame(val_rows, columns=VALUE_COLUMNS), pd.DataFrame(grp_rows, columns=GROUP_COLUMNS))
        db._clean()
        return db

    @classmethod
    def from_summary(cls, groups=None, verbose=False):
        """build the database by running the load_* functions of PDR_summary.py (all of them by default)"""
        S = _summary()
        loaders = [(f.__code__.co_firstlineno, n) for n, f in inspect.getmembers(S, inspect.isfunction)
                   if n.startswith('load_') and f.__module__ == S.__name__ and n != 'load_empty']
        samples = {}
        for _, n in sorted(loaders):
            g = n[len('load_'):]
            if groups is not None and g not in groups:
                continue
            with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
                samples[g] = getattr(S, n)()
        return cls.from_samples(samples)

    # ---------------------------------------------------------------- FITS input / output
    @classmethod
    def read(cls, filename):
        """read the database from a FITS file"""
        from astropy.io import fits
        from astropy.table import Table
        dfs = {}
        with fits.open(filename) as hdul:
            for ext in EXTENSIONS:
                t = Table.read(hdul[ext])
                dfs[ext] = _columns_to_df({c: t[c] for c in t.colnames})
        db = cls(dfs['SYSTEMS'], dfs['COMPONENTS'], dfs['VALUES'], dfs['GROUPS'])
        db._clean()
        return db

    def write(self, filename, overwrite=True):
        """write the database to a FITS file"""
        from astropy.io import fits
        from astropy.table import Table
        self._clean()
        primary = fits.PrimaryHDU()
        primary.header['CREATOR'] = ('PDR_database.py', 'PDRanalyser database of absorption systems')
        primary.header['NSYS'] = (len(self.systems), 'number of systems')
        primary.header['NGROUPS'] = (self.groups['group'].nunique(), 'number of groups')
        hdus = [primary]
        for ext, df in zip(EXTENSIONS, (self.systems, self.components, self.values, self.groups)):
            hdu = fits.table_to_hdu(Table(_df_to_columns(df)))
            hdu.name = ext
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)

    # ---------------------------------------------------------------- PDR_summary objects
    def groups_list(self):
        """names of the groups, in the order of the file"""
        return list(dict.fromkeys(self.groups['group']))

    def load(self, group):
        """PDR_systems.sample with the systems (and components) of a group, as used by PDR_exc"""
        S = _systems()
        if group not in set(self.groups['group']):
            raise KeyError(f"no group '{group}'. Groups: {self.groups_list()}")
        smp = S.sample()
        sysdf = self.systems.set_index('sys_id')
        vals = {k: v for k, v in self.values.groupby(['sys_id', 'comp'], sort=False)}
        comps = {k: v for k, v in self.components.groupby('sys_id', sort=False)}
        for r in self.groups[self.groups['group'] == group].itertuples(index=False):
            s = sysdf.loc[r.sys_id]
            q = S.qso(s['name'], s['z_em'], s['z_dla'])
            q.telescope, q.SIMBAD, q.SDSS, q.comment = s['telescope'], s['SIMBAD'], s['SDSS'], s['comment']
            q.full, q.mangastatus, q.Me_ind = s['full'], s['mangastatus'], s['Me_ind']
            q.year = int(s['year']) if s['year'].isdigit() else s['year']
            q.ref, q.progID = _split(s['ref']), _split(s['progID'])
            q.coord = json.loads(s['coord']) if s['coord'] else []
            q.m = json.loads(s['mags']) if s['mags'] else {}
            for k, v in (json.loads(s['extra']) if s['extra'] else {}).items():
                setattr(q, k, v)
            for row in vals.get((r.sys_id, -1), pd.DataFrame(columns=VALUE_COLUMNS)).to_dict('records'):
                _set_e(q, _make_e(row))
            cdf = comps.get(r.sys_id, pd.DataFrame(columns=COMPONENT_COLUMNS))
            stored = {int(c['comp']): c for c in cdf.to_dict('records')}
            use = sorted(stored) if r.comps == 'all' else ([int(x) for x in str(r.comps).split(',')] if r.comps else [])
            for i in use:
                c = stored[i]
                comp = S.sy(c['z'], c['z_err']) if np.isnan(c['z_err_minus']) else S.sy(c['z'], c['z_err'], c['z_err_minus'])
                for row in vals.get((r.sys_id, i), pd.DataFrame(columns=VALUE_COLUMNS)).to_dict('records'):
                    el = _make_e(row)
                    if row['kind'] == 'physcond':
                        comp.physcond[el.name] = el
                    else:
                        comp.e[el.name] = el
                    setattr(comp, el.name, el)
                q.comp.append(comp)
            smp[q.name] = q
        return smp

    # ---------------------------------------------------------------- pandas views
    def table(self, group=None, names=None, comp=False, errors=False):
        """
        wide pandas table of values: one row per system (comp=False) or per component (comp=True)
            - group  : group name (None = all systems)
            - names  : list of quantities, e.g. ['HI', 'H2', 'Me'] (None = all)
            - errors : add '<name>_plus', '<name>_minus' and '<name>_type' columns
        """
        v = self.values[self.values['comp'] >= 0] if comp else self.values[self.values['comp'] == -1]
        if group is not None:
            members = self.groups[self.groups['group'] == group]
            v = v[v['sys_id'].isin(members['sys_id'])]
            if comp:
                keep = set()
                for r in members.itertuples(index=False):
                    ids = self.components.loc[self.components['sys_id'] == r.sys_id, 'comp'] if r.comps == 'all' \
                        else ([int(x) for x in str(r.comps).split(',')] if r.comps else [])
                    keep |= {(r.sys_id, int(i)) for i in ids}
                mask = np.array([(s, c) in keep for s, c in zip(v['sys_id'], v['comp'])], dtype=bool)
                v = v[mask]
        if names is not None:
            v = v[v['name'].isin(names)]
        index = ['sys_id', 'comp'] if comp else ['sys_id']
        fields = ['val', 'plus', 'minus', 'type'] if errors else ['val']
        wide = v.pivot_table(index=index, columns='name', values=fields, aggfunc='first', sort=False)
        if errors:
            wide.columns = [n if f == 'val' else f'{n}_{f}' for f, n in wide.columns]
        else:
            wide.columns = [n for _, n in wide.columns]
        if names is not None:
            order = [c for n in names for c in ([n, f'{n}_plus', f'{n}_minus', f'{n}_type'] if errors else [n]) if c in wide.columns]
            wide = wide[order]
        base = self.systems.set_index('sys_id')[['name', 'z_dla']]
        if comp:
            zc = self.components.set_index(['sys_id', 'comp'])['z']
            wide.insert(0, 'z', zc.reindex(wide.index).values)
        wide = base.join(wide, how='inner') if not comp else wide.join(base, on='sys_id').set_index(['name', 'z'], append=True)
        return wide

    # ---------------------------------------------------------------- editing
    def add_system(self, sys_id, name=None, z_em=0.0, z_dla=0.0, **attrs):
        """add a new system; attrs: telescope, year, ref (list), coord (list), progID (list), SIMBAD, SDSS, comment, ..."""
        if sys_id in set(self.systems['sys_id']):
            raise ValueError(f"system '{sys_id}' already exists")
        row = {c: '' for c in SYSTEM_COLUMNS}
        row.update(sys_id=sys_id, name=name or sys_id, z_em=float(z_em), z_dla=float(z_dla))
        extra = {}
        for k, v in attrs.items():
            if k in ('ref', 'progID'):
                row[k] = _LIST_SEP.join(map(str, v if isinstance(v, (list, tuple)) else [v]))
            elif k == 'coord':
                row[k] = json.dumps(list(v))
            elif k in ('m', 'mags'):
                row['mags'] = json.dumps(v)
            elif k in SYSTEM_COLUMNS:
                row[k] = str(v)
            else:
                extra[k] = v
        row['extra'] = json.dumps(extra) if extra else ''
        self.systems = pd.concat([self.systems, pd.DataFrame([row])], ignore_index=True)
        self.set_value(sys_id, 'Me', 0.0, 0.0, 0.0, type='f')   # every qso has an 'Me' entry
        return sys_id

    def remove_system(self, sys_id):
        """remove a system with its components, values and group memberships"""
        for attr in ('systems', 'components', 'values', 'groups'):
            df = getattr(self, attr)
            setattr(self, attr, df[df['sys_id'] != sys_id].reset_index(drop=True))

    def add_component(self, sys_id, z, z_err=0.0, z_err_minus=None):
        """add a component to a system; returns its index. Groups with comps='all' include it automatically."""
        self._check_system(sys_id)
        existing = self.components.loc[self.components['sys_id'] == sys_id, 'comp']
        idx = int(existing.max()) + 1 if len(existing) else 0
        row = dict(sys_id=sys_id, comp=idx, z=float(z), z_err=float(z_err),
                   z_err_minus=np.nan if z_err_minus is None else float(z_err_minus))
        self.components = pd.concat([self.components, pd.DataFrame([row])], ignore_index=True)
        self.set_value(sys_id, 'size', 0.0, 0.0, 0.0, comp=idx, type='f')   # every sy has a 'size' entry
        return idx

    def remove_component(self, sys_id, comp):
        """remove a component and its values; groups that list it explicitly no longer include it"""
        self._check_system(sys_id)
        c = self.components
        self.components = c[~((c['sys_id'] == sys_id) & (c['comp'] == comp))].reset_index(drop=True)
        v = self.values
        self.values = v[~((v['sys_id'] == sys_id) & (v['comp'] == comp))].reset_index(drop=True)
        for i in self.groups.index[(self.groups['sys_id'] == sys_id) & (self.groups['comps'] != 'all')]:
            keep = [x for x in str(self.groups.at[i, 'comps']).split(',') if x and int(x) != comp]
            self.groups.at[i, 'comps'] = ','.join(keep)

    def set_value(self, sys_id, name, val, plus=0.0, minus=None, comp=-1, type=None, format='log',
                  b=None, kind='e', f='', ref=''):
        """
        add or change a value
            - comp   : -1 for the whole system, otherwise the component index
            - type   : 'm' measurement, 'u' upper limit, 'l' lower limit, 'f' fixed;
                       default: 'm' if there is an error, otherwise 'f'
            - format : 'log' or 'dec'
            - b      : Doppler parameter (val, plus, minus)
        """
        self._check_system(sys_id)
        if comp >= 0 and not ((self.components['sys_id'] == sys_id) & (self.components['comp'] == comp)).any():
            raise KeyError(f'system {sys_id} has no component {comp}')
        minus = plus if minus is None else minus
        if type is None:
            type = 'f' if plus == 0 and minus == 0 else 'm'
        row = dict(sys_id=sys_id, comp=int(comp), kind=kind, name=name, val=float(val), plus=float(plus),
                   minus=float(minus), type=type, format=format, b_val=np.nan, b_plus=np.nan, b_minus=np.nan,
                   b_type='', f=f, ref=ref)
        if b is not None:
            bp, bm = (b[1], b[2]) if len(b) > 2 else (b[1], b[1])
            row.update(b_val=float(b[0]), b_plus=float(bp), b_minus=float(bm), b_type='f' if bp == 0 and bm == 0 else 'm')
        mask = self._value_mask(sys_id, name, comp, kind)
        if mask.any():
            i = mask.idxmax()
            for k, v in row.items():
                self.values.at[i, k] = v
        else:
            self.values = pd.concat([self.values, pd.DataFrame([row])], ignore_index=True)

    def remove_value(self, sys_id, name, comp=-1, kind='e'):
        self.values = self.values[~self._value_mask(sys_id, name, comp, kind)].reset_index(drop=True)

    def add_group(self, group, systems=(), comps='all'):
        """create a group from a list of sys_id (comps: 'all' or list of component indices, used for every system)"""
        if group in set(self.groups['group']):
            raise ValueError(f"group '{group}' already exists (use add_to_group)")
        for s in systems:
            self.add_to_group(group, s, comps)

    def add_to_group(self, group, sys_id, comps='all'):
        self._check_system(sys_id)
        name = self.systems.loc[self.systems['sys_id'] == sys_id, 'name'].iloc[0]
        members = self.groups[self.groups['group'] == group]
        clash = self.systems.set_index('sys_id').loc[members['sys_id'], 'name']
        if sys_id in set(members['sys_id']):
            raise ValueError(f"'{sys_id}' is already in group '{group}'")
        if name in set(clash):
            raise ValueError(f"group '{group}' already has a system named '{name}'")
        comps = comps if isinstance(comps, str) else ','.join(str(int(c)) for c in comps)
        self.groups = pd.concat([self.groups, pd.DataFrame([dict(group=group, sys_id=sys_id, comps=comps)])], ignore_index=True)

    def remove_from_group(self, group, sys_id):
        self.groups = self.groups[~((self.groups['group'] == group) & (self.groups['sys_id'] == sys_id))].reset_index(drop=True)

    def remove_group(self, group):
        self.groups = self.groups[self.groups['group'] != group].reset_index(drop=True)

    def rename_group(self, old, new):
        self.groups.loc[self.groups['group'] == old, 'group'] = new

    def subset(self, groups):
        """
        new database with only the given groups: their systems, the components the groups use,
        the values of those systems and components, and only these group memberships
        """
        groups = [groups] if isinstance(groups, str) else list(groups)
        missing = [g for g in groups if g not in set(self.groups['group'])]
        if missing:
            raise KeyError(f'no groups {missing}')
        grp = self.groups[self.groups['group'].isin(groups)].reset_index(drop=True)
        keep_sys = set(grp['sys_id'])
        keep_comp = set()
        for r in grp.itertuples(index=False):
            ids = self.components.loc[self.components['sys_id'] == r.sys_id, 'comp'] if r.comps == 'all' \
                else ([int(x) for x in str(r.comps).split(',')] if r.comps else [])
            keep_comp |= {(r.sys_id, int(i)) for i in ids}
        sysdf = self.systems[self.systems['sys_id'].isin(keep_sys)]
        comps = self.components[[(s, c) in keep_comp for s, c in zip(self.components['sys_id'], self.components['comp'])]]
        v = self.values
        vals = v[[s in keep_sys and (c < 0 or (s, c) in keep_comp) for s, c in zip(v['sys_id'], v['comp'])]]
        sub = PDRdatabase(sysdf.copy(), comps.copy(), vals.copy(), grp.copy())
        sub._clean()
        return sub

    def find(self, name):
        """sys_id of systems whose name contains the given string"""
        return list(self.systems.loc[self.systems['name'].str.contains(name, regex=False), 'sys_id'])

    # ---------------------------------------------------------------- helpers
    def _check_system(self, sys_id):
        if sys_id not in set(self.systems['sys_id']):
            raise KeyError(f"no system '{sys_id}'")

    def _value_mask(self, sys_id, name, comp, kind):
        v = self.values
        return (v['sys_id'] == sys_id) & (v['name'] == name) & (v['comp'] == comp) & (v['kind'] == kind)

    def _clean(self):
        """consistent column types (str columns without NaN, int component indices)"""
        for attr, cols in (('systems', SYSTEM_COLUMNS), ('components', COMPONENT_COLUMNS),
                           ('values', VALUE_COLUMNS), ('groups', GROUP_COLUMNS)):
            df = getattr(self, attr).reindex(columns=cols).reset_index(drop=True)
            for c in cols:
                if c in _STR_COLUMNS:
                    df[c] = df[c].fillna('').astype(str).str.rstrip().astype(object)   # FITS drops trailing spaces
                elif c in _INT_COLUMNS:
                    df[c] = df[c].astype(int)
                else:
                    df[c] = df[c].astype(float)
            setattr(self, attr, df)

    def __repr__(self):
        return (f'PDRdatabase: {len(self.systems)} systems, {len(self.components)} components, '
                f'{len(self.values)} values, groups: {self.groups_list()}')


def _set_e(obj, el):
    obj.e[el.name] = el
    setattr(obj, el.name, el)


# =====================================================================================
# DataFrame <-> FITS columns
# =====================================================================================

def _df_to_columns(df):
    """dict of numpy arrays for astropy.table.Table (fixed-width strings, int64, float64)"""
    cols = {}
    for c in df.columns:
        if c in _STR_COLUMNS:
            bad = [(i, x) for i, x in zip(df.index, df[c]) if not str(x).isascii()]
            if bad:
                raise ValueError(f"FITS text can only contain ASCII characters; column '{c}', rows {[i for i, _ in bad[:5]]}: "
                                 f"{[x for _, x in bad[:5]]}")
            arr = np.array([str(x) for x in df[c]], dtype=str)
            if arr.size == 0 or arr.dtype.itemsize == 0:
                arr = arr.astype('U1')
            cols[c] = arr
        elif c in _INT_COLUMNS:
            cols[c] = np.asarray(df[c], dtype=np.int64)
        else:
            cols[c] = np.asarray(df[c], dtype=np.float64)
    return cols


def _columns_to_df(cols):
    """DataFrame from columns read from FITS (bytes -> str, big-endian -> native, masked -> filled)"""
    out = {}
    for c, col in cols.items():
        arr = col.filled() if hasattr(col, 'filled') else col
        arr = np.asarray(arr)
        if arr.dtype.kind == 'S':
            arr = np.char.decode(arr, 'utf-8')
        if arr.dtype.kind == 'U':
            out[c] = pd.Series([str(x).rstrip() for x in arr], dtype=object)
        else:
            out[c] = pd.Series(arr.astype(arr.dtype.newbyteorder('=')))
    return pd.DataFrame(out)


# =====================================================================================
# check
# =====================================================================================

def compare_samples(s1, s2):
    """list of differences between two PDR_summary samples (empty list = identical)"""
    diffs = []

    def cmp_e(where, d1, d2):
        if list(d1) != list(d2):
            diffs.append(f'{where}: entries {list(d1)} != {list(d2)}')
        for k in set(d1) & set(d2):
            x, y = d1[k], d2[k]
            for at in ('val', 'plus', 'minus', 'type', 'repr', 'default_format'):
                if getattr(x.col, at) != getattr(y.col, at):
                    diffs.append(f'{where} {k}.{at}: {getattr(x.col, at)} != {getattr(y.col, at)}')
            if (x.b is None) != (y.b is None) or (x.b is not None and (x.b.val, x.b.plus, x.b.minus, x.b.type) != (y.b.val, y.b.plus, y.b.minus, y.b.type)):
                diffs.append(f'{where} {k}.b differs')
            for at in ('f', 'ref', 'J', 'nu', 'name'):
                if getattr(x, at, None) != getattr(y, at, None):
                    diffs.append(f'{where} {k}.{at}: {getattr(x, at, None)} != {getattr(y, at, None)}')

    if list(s1) != list(s2):
        diffs.append(f'systems/order differ: {list(s1)[:5]}... vs {list(s2)[:5]}...')
    for name in set(s1) & set(s2):
        q1, q2 = s1[name], s2[name]
        for at in ('name', 'z_em', 'z_dla', 'telescope', 'year', 'coord', 'progID', 'SIMBAD', 'SDSS', 'comment',
                   'full', 'mangastatus', 'Me_ind', 'm'):
            v1, v2 = getattr(q1, at), getattr(q2, at)
            if isinstance(v1, str) and isinstance(v2, str):
                v1, v2 = v1.rstrip(), v2.rstrip()   # trailing spaces are not kept in FITS
            if v1 != v2:
                diffs.append(f'{name}.{at}: {getattr(q1, at)!r} != {getattr(q2, at)!r}')
        r1 = q1.ref if isinstance(q1.ref, list) else ([q1.ref] if q1.ref else [])
        if r1 != q2.ref:
            diffs.append(f'{name}.ref: {q1.ref!r} != {q2.ref!r}')
        x1 = {k: v for k, v in vars(q1).items() if k not in _QSO_STANDARD and not isinstance(v, e)}
        x2 = {k: v for k, v in vars(q2).items() if k not in _QSO_STANDARD and not isinstance(v, e)}
        if json.loads(json.dumps(x1)) != x2:
            diffs.append(f'{name} extra attributes differ: {x1} != {x2}')
        cmp_e(name, q1.e, q2.e)
        if len(q1.comp) != len(q2.comp):
            diffs.append(f'{name}: {len(q1.comp)} components != {len(q2.comp)}')
        for i, (c1, c2) in enumerate(zip(q1.comp, q2.comp)):
            if c1.z != c2.z or (c1.z_err if not isinstance(c1.z_err, tuple) else tuple(c1.z_err)) != c2.z_err:
                diffs.append(f'{name} comp {i}: z {c1.z}/{c1.z_err} != {c2.z}/{c2.z_err}')
            cmp_e(f'{name} comp {i}', c1.e, c2.e)
            cmp_e(f'{name} comp {i} physcond', c1.physcond, c2.physcond)
    return diffs


def selftest(db, verbose=True):
    """compare every group of the database with PDR_summary.py; returns the number of differences"""
    S = _summary()
    n = 0
    for g in db.groups_list():
        with contextlib.redirect_stdout(io.StringIO()):
            orig = getattr(S, 'load_' + g)() if hasattr(S, 'load_' + g) else None
        if orig is None:
            if verbose:
                print(f'{g:15s} (not in PDR_summary.py, skipped)')
            continue
        d = compare_samples(orig, db.load(g))
        n += len(d)
        if verbose:
            print(f'{g:15s} {len(orig):4d} systems  ' + ('OK' if not d else f'{len(d)} differences, e.g. {d[:3]}'))
    return n


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    fname = sys.argv[2] if len(sys.argv) > 2 else 'PDR_database_full.fits'
    if cmd == 'export':
        db = PDRdatabase.from_summary()
        db.write(fname)
        print(f'written {fname}: {db}')
    elif cmd == 'selftest':
        db = PDRdatabase.read(fname)
        print(db)
        n = selftest(db)
        print('all groups identical to PDR_summary.py' if n == 0 else f'{n} differences')
    else:
        print(__doc__)
