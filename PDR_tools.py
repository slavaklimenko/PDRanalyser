"""
PDR_tools.py

Minimal, self-contained subset of the `spectro` package (S. Balashev) needed by
PDR_exc.py, PDRviewer.py, PDR_systems.py and pyratio.py, so the PDR code runs without spectro.

Contents
    a          value with asymmetric uncertainties        (spectro.a_unc)
    distr      helper distribution used by `a`            (spectro.a_unc)
    distr1d    1D likelihood distribution                 (spectro.stats)
    distr2d    2D likelihood distribution                 (spectro.stats)
    e          species / column density container         (spectro.atomic)
    load_pickle  load pickles written with the old module names (spectro.*, H2_exc, H2_summary)

Level populations (spectro.pyratio) are in the separate file pyratio.py, kept identical
to spectro except for its import block.
"""

import copy
import pickle
import re
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import interpolate, integrate, optimize
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, norm, rv_continuous


# =====================================================================================
# compatibility with new NumPy / SciPy
# (scipy.integrate.simps and scipy.interpolate.interp2d were removed in SciPy 1.14;
#  on older versions the original functions are used, so results are unchanged)
# =====================================================================================

def _simps(y, x, axis=-1):
    if hasattr(integrate, 'simps'):
        return integrate.simps(y, x, axis=axis)
    return integrate.simpson(y, x=x, axis=axis)


class _Interp2dGrid:
    """replacement for interp2d on a regular grid: f(x, y) -> array (len(y), len(x)), fill_value outside the grid"""
    def __init__(self, x, y, z, kind='linear', fill_value=None):
        k = {'linear': 1, 'cubic': 3, 'quintic': 5}[kind]
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.spl = interpolate.RectBivariateSpline(self.y, self.x, np.asarray(z, dtype=float), kx=k, ky=k, s=0)
        self.fill_value = fill_value

    def __call__(self, x, y):
        x, y = np.sort(np.atleast_1d(np.asarray(x, dtype=float))), np.sort(np.atleast_1d(np.asarray(y, dtype=float)))
        z = self.spl(y, x)
        if self.fill_value is not None:
            z[(y < self.y[0]) | (y > self.y[-1]), :] = self.fill_value
            z[:, (x < self.x[0]) | (x > self.x[-1])] = self.fill_value
        return z[0] if len(z) == 1 else z


def _interp2d(x, y, z, kind='linear', fill_value=None):
    try:
        return interpolate.interp2d(x, y, z, kind=kind, fill_value=fill_value)
    except (AttributeError, NotImplementedError):
        return _Interp2dGrid(x, y, z, kind=kind, fill_value=fill_value)



# =====================================================================================
# asymmetric uncertainties  (from spectro/a_unc.py)
# =====================================================================================

class a:
    r"""
    This class provide basic arithmetic operations with the values
    in case of asymmetric uncertainties.
    
    Written by Serj Balashev
    email: s.balashev@gmail.com
    
    To import class put a_unc folder in python path folder and just use:
    #    >>> from a_unc import a
        
    Uncertainties are estimated by minimization of joint likelihood function
    for details see e.g. Barlow 2003 - arXiv:physics/0406120
    
    Supported operations are: +, -, *, /, product by some factor
    
    You can initialize asymmetric value by several ways:
    1. set three values and additional parameters:
        x = a(19, 0.3, 0.2, form)
        x = a(19, 0.2, form)  -- will be symmetric value
             
                 form is optional and set representation format:
                        1. 'd' or 'dec' - for decimal
                        2. 'l' or 'log' - for log10
                        3. if not set - set to be default
                            
    2. using latex style 
        x = a('$19.0^{+0.3}_{-0.2}$', form)
        x = a('$19.0\pm0.2$', form)
        
    3. using list 
        x = a([19.0, 0.3, 0.2], form)
    
    Upper and lower limits also can be specified:
    1. manually:
        x = a(19, form, t='')
                
                where <t> specified limit:
                    t='u'   : for upper limits
                    t='l'   : for lower limits
    2. using latex style:
        x = a('$<19.0$', form)
        
    Example of usage: 
    
    x = a(19,0.3,0.2)
    y = a(2,0.5,0.5,'d')
    z = x * y
    z.log()
    print(z)
    z.plot_lnL() 
    
    
    Some useful methods:
    a.log() - convert to log10 format
    a.dec() - convert to decimal format
    a.default() - convert to default format set by <default_format> variable
    a.latex() - convert to latex format output
    a.lnL(x) - return the ln from Likelihood function at x position
    a.plot_lnL(x) - plot ln of Likelihood function
    
    Parameters:
    default_format - default format for input
    likelihood_approx - type of approximation of asymmetric likelyhood function
                        see lnL method and Barlow 2003 (arXiv:physics/0406120)
    """
    
    def __init__(self, *args, **kwargs):

        self.likelihood_approx = 2
        self.method = 'Nelder-Mead'
        self.method = 'L-BFGS-B'
        #self.method = 'SLSQP'
        #self.method = 'TNC'
        self.tol = 1e-3

        # if the input is empty       
        if len(args) in [0]:
            self.val = 0
            self.plus = 0
            self.minus = 0
            self.type = 'f'
            
        if len(args) in [1, 2]:
            # if the input like latex style a($19.00^{+0.10}_{-0.10}$, 'l')        
            if isinstance(args[0], str):
                self.fromtex(args[0])
            # if the input like latex style a([19.00, 0.01, 0.01], 'l')
            elif isinstance(args[0], (list, tuple)):
                self.val = float(args[0][0])
                self.plus = float(args[0][1])
                self.minus = float(args[0][2])
                if self.plus == 0 and self.minus == 0:
                    self.type = 'f'
                else:
                    self.type = 'm'
            elif isinstance(args[0], (float, int, np.floating)):
                self.val = float(args[0])
                if len(args) > 1 and isinstance(args[1], (float, int)):
                    self.plus = float(args[1])
                    self.minus = float(args[1])
                    self.type = 'm'
                else:
                    self.plus = 0
                    self.minus = 0
                    self.type = 'f'
            
                
        # if the input like data style a(19.00, 0.10, 0.10, 'l')
        elif len(args) in [3, 4]:
            self.val = float(args[0])
            self.plus = abs(float(args[1]))
            if isinstance(args[2], (int, float, np.floating)):
                self.minus = abs(float(args[2]))
            elif isinstance(args[2], (str)):
                self.minus = abs(float(args[1]))
            if self.plus == 0 and self.minus == 0:
                self.type = 'f'
            else:
                self.type = 'm'


        
        if 't' in kwargs.keys():
            self.type = kwargs['t']

        if any(x in args for x in ['d', 'dec', 'f']):
            self.repr = 'dec'
            self.default_format = self.repr
        elif any(x in args for x in ['l', 'log']):
            self.repr = 'log'
        else:
            #if self.plus < 0.9 and self.minus < 0.9:
            #    self.repr = 'log'
            self.repr = 'log'
            #else:
            #self.repr = 'dec'

        if 'f' in kwargs.keys():
            if kwargs['f'] in ['d', 'dec']:
                self.repr = 'dec'
            if kwargs['f'] in ['l', 'log']:
                self.repr = 'log'
        self.default_format = self.repr

        #if any(x in kwargs for x in ['d', 'dec', 'f']):
        #    self.repr = 'dec'
        #    self.default_format = self.repr
        #elif any(x in kwargs for x in ['l', 'log']):
        #    self.repr = 'log'
        #else:
            #if self.plus < 0.9 and self.minus < 0.9:
        #    #    self.repr = 'log'
        #    self.repr = 'log'
            #else:
            #self.repr = 'dec'
        #self.default_format = self.repr

    def __str__(self):
        if self.repr == 'log':
            if self.type in ['m', 'f']:
                return "{0:.2f} +{1:.2f} -{2:.2f} in {3} format".format(self.val, self.plus, self.minus, self.repr)
            elif self.type == 'u':
                return "<{0:.2f} in {1} format".format(self.val, self.repr)
            elif self.type == 'l':
                return ">{0:.2f} in {1} format".format(self.val, self.repr)
                
        if self.repr == 'dec':
            if self.type in ['m', 'f']:
                return "{0:.2e} +{1:.2e} -{2:.2e} in {3} format".format(self.val, self.plus, self.minus, self.repr)
            elif self.type == 'u':
                return "<{0:.2e} in {1} format".format(self.val, self.repr)
            elif self.type == 'l':
                return ">{0:.2e} in {1} format".format(self.val, self.repr)
    
    def __repr__(self):
        return self.__str__()
    
    def latex(self, eqs=1, f=2, base=None):
        """
        show representation of <a> values in tex
        parameters:
                
                eqs      : if ==1 add $...$
                f        : format - number of numbers after floating point
                base     : base for dec values                
        """
        if np.isfinite(self.plus) and np.isfinite(self.minus):
            if self.repr == 'log':
                #print(self.val, self.plus, self.minus, f)
                s = "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(self.val, self.plus, self.minus, n=f)
            if self.repr == 'dec':
                if base == None:
                    base = int(np.log10(abs(self.val)))
                if base == 0:
                    s = "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(self.val/10**base, self.plus/10**base, self.minus/10**base, n=f)
                else:
                    s = "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}} \\times 10^{{{b}}}".format(self.val/10**base, self.plus/10**base, self.minus/10**base, b=base, n=f)
        else:
            if not np.isfinite(self.plus) and np.isfinite(self.minus):
                pre, val = '>', self.val - self.minus
            else:
                pre, val = '<', self.val + self.plus
            if self.repr == 'log':
                #print(self.val, self.plus, self.minus, f)
                s = "{0:s}{1:.{n}f}".format(pre, val, n=f)
            if self.repr == 'dec':
                if base == None:
                    base = int(np.log10(abs(val)))
                if base == 0:
                    s = "{0:s}{1:.{n}f}".format(pre, val/10**base, n=f)
                else:
                    s = "{0:s}{1:.{n}f} \\times 10^{{{b}}}".format(pre, val/10**base, b=base, n=f)

        la = '$' if eqs == 1 else ''
        return la + s + la
        
    def fromtex(self, text):
        """
        get <a> object from latex expression as 
        "$15.75^{+0.06}_{-0.07}$"
        "$15.75\pm0.07$"
        "$<15.75$"
        parameter:
            - text     : latex expression
                         type: str
        """
        text = text.replace('$', '')
        if text.find('pm') > -1:
            self.val = float(text[:text.find(r'\pm')])
            self.plus = float(text[text.find(r'\pm')+3:])
            self.minus = self.plus
            self.type = 'm'
        elif text.find('<') > -1:
            self.val = float(text.split('<')[1])
            self.plus = 0
            self.minus = 0
            self.type = 'u'
        elif text.find('>') > -1:
            self.val = float(text.split('>')[1])
            self.plus = 0
            self.minus = 0
            self.type = 'l'
        elif text.find('(') > -1 and text.find(')') > -1:
            text = text.replace('(', '').replace(')', '')
            self.val = float(text.split('^')[0])
            self.plus = abs(float(re.findall(r'\^\{([^}]*)\}', text)[0]))/10**(len(text.split('^')[0])-2)
            self.minus = abs(float(re.findall(r'_\{([^}]*)\}', text)[0]))/10**(len(text.split('^')[0])-2)
            self.type = 'm'
        else:
            self.val = float(text.split('^')[0])
            self.plus = abs(float(re.findall(r'\^\{([^}]*)\}', text)[0]))
            self.minus = abs(float(re.findall(r'_\{([^}]*)\}', text)[0]))
            self.type = 'm'
        #print(self.val, self.plus, self.minus)

    def check_type(self):
        if self.type == 'm' and self.plus == 0 and self.minus == 0:
            self.type = 'f'
        if self.type == 'f' and (self.plus != 0 or self.minus != 0):
            self.type = 'm'

    def dec(self):
        if self.repr == 'log':
            if self.type in ['m', 'f']:
                self.plus = 10**(self.val+self.plus) - 10**self.val
                self.minus = 10**self.val - 10**(self.val-self.minus)
            self.val = 10**self.val
            self.repr = 'dec'
        return self
                
    def log(self):
        if self.repr == 'dec':
            #assert self.val > 0 and self.val-self.minus > 0, ('Argument of log is <= 0')
            if self.type in ['m', 'f']:
                self.plus = np.log10(self.val+self.plus) - np.log10(self.val)
                if self.val < 0:
                    assert self.val > 0, ('Argument of log is <= 0')
                elif self.val-self.minus < 0:
                    self.minus = np.inf
                    warnings.warn('Argument of minus boundary is <= 0, set to -np.inf')
                else:
                    self.minus = np.log10(self.val) - np.log10(self.val - self.minus)
            self.val = np.log10(self.val)
            self.repr = 'log'
        return self
    
    def default(self, fmt=None):
        """
        Print (if fmt in not specified) or set (if fmt is specified) default format
        """
        self.check_type()
        if fmt is None:
            if self.default_format == 'log':
                return self.log()
            if self.default_format == 'dec':
                return self.dec()
        else:
            if fmt in ['log', 'l']:
                self.default_format == 'log'
            elif fmt in ['dec', 'd']:
                self.default_format == 'dec'

    def copy(self):
        return copy.deepcopy(self)

    def __add__(self, other):
        
        self.dec()
        #res = A_unc(0,0,0,typ='d')
        res = copy.deepcopy(self)

        if isinstance(other, a):
            other.dec()
            res.val, res.plus, res.minus = self.val + other.val, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2)
            if self.type == 'f' or other.type == 'f':
                pass
            if self.type == 'm' and other.type == 'm':
                res = self.mini(self.sum_lnL, other, res)
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val + other
            res.plus = self.plus
            res.minus = self.minus
        return res.default()
    
    def __radd__(self, other):
        
        res = self.__add__(other)
        
        return res.default()
        
    def __sub__(self, other):
        
        self.dec()
        #res = A_unc(0,0,0,typ='d')
        res = copy.deepcopy(self)

        if isinstance(other, a):
            other.dec()
            #print(self.plus, other.plus, self.minus, other.minus, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2))
            res.val, res.plus, res_minus = self.val - other.val, np.sqrt(self.plus ** 2 + other.plus ** 2), np.sqrt(self.minus ** 2 + other.minus ** 2)
            #print(res.val, res.plus, res_minus)
            res = self.mini(self.sub_lnL, other, res)
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val - other
            res.plus = self.plus
            res.minus = self.minus
        return res.dec()
        
    def __truediv__(self, other):
        
        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')

        if isinstance(other, a):
            other.dec()
            if self.type == 'm' and other.type == 'm':
                res.val = self.val / other.val
                res.val, res.plus, res.minus = self.val / other.val, np.sqrt((self.plus / other.val)**2 + (self.val / other.val**2 * other.plus)**2), np.sqrt((self.minus / other.val)**2 + (self.val / other.val**2 * other.minus)**2)
                #if (res.minus > res.value): res.minus = res.value * 0.9
                res = self.mini(self.div_lnL, other, res)
            else: 
                res.val = self.val/other.val
                
                if other.type == 'f':
                    res.plus, res.minus = self.plus/other.val, self.minus/other.val 
                if other.type == 'u':
                    if self.type == 'u':
                        raise TypeError('Value is unconstrained: upper limit is divided by upper limit')
                    else:
                        res.type = 'l'
                        
                elif other.type == 'l':
                    if self.type == 'l':
                        raise TypeError('Value is unconstrained: lower limit is divided by lower limit')
                    else:
                        res.type = 'u'
                if res.type in ['u', 'l']:
                    res.plus, res.minus = 0, 0
            other.default()
                
        elif isinstance(other, (int, float)):
            res.val = self.val / other
            res.plus = self.plus / other
            res.minus = self.minus / other
        
        return res.default()
    
    def __rtruediv__(self, other):
        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')
        if isinstance(other, (int, float)):
            res.val = other / self.val
            if self.val - self.minus == 0:
                res.plus = res.val*1e10
            else:
                res.plus = abs(other / (self.val - self.minus) - res.val)
            if (self.val + self.plus) == 0:
                res.minus = res.val*1e10
            else:
                res.minus = abs(res.val - other / (self.val + self.plus))
            if self.val < 0:
                res.plus, res.minus = res.minus, res.plus
        return res.default()
        
    def __mul__(self, other):

        self.dec()
        res = copy.deepcopy(self)
        #res = A_unc(0,0,0,typ='d')

        if isinstance(other, a):

            other.dec()
            if self.type == 'm' and other.type == 'm':
                res.val, res.plus, res.minus = self.val * other.val,  np.sqrt(self.val**2 * other.plus**2 + self.plus**2 * other.val**2), np.sqrt(self.val**2 * other.minus**2 + self.minus**2 * other.val**2)
                #print(res.val, res.plus, res.minus)
                res = self.mini(self.mul_lnL, other, res)
            else: 
                res.val = self.val*other.val
                if (self.type == 'l' and other.type == 'u') or (self.type == 'u' and other.type == 'l'):
                    raise TypeError('Value is unconstrained: lower limit is multiplied by upper limit')
                else:
                    if other.type != 'm':
                       res.type = other.type
                if res.type in ['u', 'l']:
                    res.plus, res.minus = 0, 0
            other.default()

        elif isinstance(other, (int, float)):
            res.val = self.val * other
            res.plus = self.plus * other
            res.minus = self.minus * other
        
        return res.default()

    def minim(self, delta, func, other):
        m = minimize(func, 1, args=(other, delta), method=self.method, bounds=[(self.tol, None)], options={'ftol':self.tol, 'eps':0.001})
        #print(delta, m.fun, m.x)
        return np.abs(m.fun - 0.5)

    def mini(self, func, other, res):
        #print((minimize(self.minim, 1 + res.plus / 2 / res.val, args=(func, other), method=self.method,
        #                bounds=[(0.999, None)], options={'ftol': self.tol, 'eps': 0.001}).x[0] - 1))
        res.plus = res.val * (minimize(self.minim, 1 + res.plus / 2 / res.val, args=(func, other), method=self.method, bounds=[(0.999, None)], options={'ftol': self.tol, 'eps': 0.001}).x[0] - 1)
        res.minus =res.val * (1 - minimize(self.minim, 1 - res.minus / 2 / res.val, args=(func, other), method=self.method, bounds=[(None, 0.99)], options={'ftol': self.tol, 'eps': 0.001}).x[0])
        #print(self.minim, 1 - res.minus / 2 / res.val, minimize(self.minim, 1 - res.minus / 2 / res.val, args=(func, other), method=self.method, bounds=[(None, 0.9)], options={'ftol': self.tol, 'eps': 0.001}).x[0])
        return res

    def sum_lnL(self, x, other, delta):
        return (-1)*self.lnL(delta * (self.val + other.val) - x * other.val) - other.lnL(x * other.val)
    
    def sub_lnL(self, x, other, delta):
        return (-1)*self.lnL(delta * (self.val - other.val) + x * other.val) - other.lnL(x * other.val)
    
    def mul_lnL(self, x, other, delta):
        return (-1)*self.lnL(delta * self.val / x) - other.lnL(x * other.val)
    
    def div_lnL(self, x, other, delta):
        return (-1)*self.lnL(delta * self.val * x) - other.lnL(x * other.val)
        
    def lnL(self, x, ind=2):
        alpha = x - self.val
        if self.type == 'm':
            if self.plus != self.minus:
                # Barlow 2003
                if ind == 1:
                    beta = self.plus / self.minus
                    gamma = self.plus * self.minus / (self.plus + self.minus)
                    return -0.5 * (np.log(1 + alpha / gamma) / np.log(beta)) ** 2
                # two parabola
                if ind == 2:
                    if alpha > 0:
                        return -0.5 * (alpha / self.plus) ** 2
                    else:
                        return -0.5 * (alpha / self.minus) ** 2
                # Another from Barlow arXiv:physics/0406120
                if ind == 3:
                    return -0.5 * (alpha * (self.plus + self.minus) / (2 * self.plus * self.minus + (self.plus - self.minus) * alpha)) ** 2
                # Another from Barlow arXiv:physics/0406120
                if ind == 4:
                    return -0.5 * alpha**2 / (self.plus * self.minus + (self.plus - self.minus) * alpha)
            else:
                #print (((x-n[0])/n[1])**2)
                return -0.5 * (alpha / self.plus) ** 2

        elif self.type in ['l', 'u']:
            d = {'l': -1, 'u': 1}
            return np.log(np.abs((np.sign(alpha) - d[self.type]) / 2))



    def L(self, x, ind=2):
        return np.exp(self.lnL(x, ind=ind))

    def plot_lnL(self):
        self.default()
        z = np.linspace(self.val-3*self.minus,self.val+3*self.plus,100)
        y = [self.lnL(z) for z in z]
        plt.plot(z,y)
        plt.axhline(-0.5, c='k', ls='--')

    def rvs(self, n=1, repr='log'):
        """
        Generates the sample drawn from the likelihood related to <a> object
        parameters:
            - n          :  length of the sample
            - repr       :  log or dec (for generating pdf)
        return:
            - sample     :  1d np.array with the sample of n
        """
        getattr(self, repr)()
        d = distr(self, a=self.val - 4 * self.minus, b=self.val + 4 * self.plus)

        return d.rvs(size=n)


class distr(rv_continuous):

    def __init__(self, y, a, b):
        super().__init__(a=a, b=b)
        self.y = y
        self.norm = quad(self.y.L, a=a, b=b)

    def _pdf(self, x):
        return 1 / self.norm[0] * self.y.L(x)


# =====================================================================================
# 1D / 2D likelihood distributions  (from spectro/stats.py)
# =====================================================================================

class distr1d():
    """
    Dealing with 1d distributions.

    Example of usage:
    d = distr1d(x, y), where x and y array specified distribution

    Make point estimate:
    d.dopoint()

    Make interval estimate:
    d.dointerval(conf=0.683)

    Make plot:
    d.plot(conf=0.683)
    """
    def __init__(self, x, y=None, xtol=1e-5, debug=False, name='par',mode='int'):
        self.x = x
        if y is not None:
            self.y = y
        else:
            self.kde()
        self.xtol = xtol
        self.debug = debug
        self.name = name
        self.normalize(mode=mode)
        self.interpolate()

    def kde(self):
        kde = gaussian_kde(self.x)
        x = np.linspace(np.min(self.x), np.max(self.x), np.max([100, int(np.sqrt(len(self.x)))]))
        self.y = kde(x)
        self.x = x

    def normalize(self,mode='int'):
        if mode == 'int':
            inter = interpolate.interp1d(self.x, self.y)
            norm = integrate.quad(inter, self.x[0], self.x[-1])
            self.y = self.y / norm[0]
            self.ymax = np.max(self.y)
        elif mode == 'max':
            self.ymax = np.max(self.y)
            self.y = self.y / self.ymax
            self.ymax = np.max(self.y)

    def normaliz(self):
        inter = interpolate.interp1d(self.x, self.y)
        norm = integrate.quad(inter, self.x[0], self.x[-1])
        self.y = self.y / norm[0]
        self.ymax = np.max(self.y)

    def interpolate(self, kind='linear'):
        self.inter = interpolate.interp1d(self.x, self.y, kind=kind, bounds_error=False, fill_value=0)

    def minter(self, x):
        return -self.inter(x)

    def level(self, x, level):
        return np.squeeze(self.inter(x) - level)[()]  # scalar for root finders (NumPy >= 2)

    def plot(self, x=None, conf=None, kind='center', color='orangered', ax=None, xlabel=None, ylabel=None, fontsize=16, alpha=0.3,label=None):
        if ax is None:
            fig, ax = plt.subplots()
        if x is None:
            x = self.x
        if label == None:
            ax.plot(x, self.inter(x), '-', color=color, lw=1.5)
        else:
            ax.plot(x, self.inter(x), '-', color=color, lw=1.5,label=label)
        if conf is not None:
            self.dointerval(conf=conf, kind=kind)
            print('interval plot', self.interval)
            mask = np.logical_and(x >= self.interval[0], x <= self.interval[1])
            ax.fill_between(x[mask], self.inter(x)[mask], facecolor=color, alpha=alpha, interpolate=True)

        ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontsize)

        if ylabel is None:
            ax.set_ylabel('pdf', fontsize=fontsize)
        else:
            ax.set_ylabel(ylabel, fontsize=fontsize)

        return ax

    def minmax(self, level):
        ind = np.argmax(self.y)
        if level > 0 and level < self.ymax:
            if 1:
                try:
                    self.xmin = optimize.bisect(self.level, self.x[0], self.x[ind], args=(level,))
                except ValueError:
                    self.xmin = self.x[0]
                try:
                    self.xmax = optimize.bisect(self.level, self.x[ind], self.x[-1], args=(level,))
                except ValueError:
                    self.xmax = self.x[-1]
            else:
                step = (self.x[ind + 1] - self.x[ind - 1]) / 10
                o = {'xtol': self.xtol, 'epsfcn': step}
                self.xmin = optimize.root(self.level, self.x[ind-3], args=(level,), method='lm', disp=self.debug, options=o).x
                self.xmax = optimize.root(self.level, self.x[ind+3], args=(level,), method='lm', disp=self.debug, options=o).x
        elif level > self.ymax:
            self.xmin, self.xmax = self.point, self.point
        elif level <= 0:
            self.xmin, self.xmax = self.x[0], self.x[-1]

        return self.xmin, self.xmax

    def func_twoside(self, level, conf):
        xmin, xmax = self.minmax(level)
        return integrate.quad(self.inter, xmin, xmax)[0] - conf

    def func_oneside(self, x, conf, kind='left'):
        if kind == 'right':
            return integrate.quad(self.inter, x, self.x[-1])[0] - conf
        elif kind == 'left':
            return integrate.quad(self.inter, self.x[0], x)[0] - conf

    def dopoint(self, verbose=True):
        """
        Point estimate of distribution
        :return:
            - point estimate?
        """
        self.point = optimize.fmin(self.minter, self.x[np.argmax(self.y)], xtol=self.xtol, disp=self.debug)[0]
        self.ymax = self.inter(self.point)
        if self.debug or verbose:
            print('point:', self.point, self.x[np.argmax(self.y)])
        return self.point

    def dointerval(self, conf=0.683, kind='center', verbose=False):
        """
        Interval estimate for given confidence level
        parameters:
            - conf        :  confidence level
            - kind        :  type of the interval, can be: 'center' (default), 'left', 'right'
            - verbose     :  verbose output
        return: interval, level
            - interval    :  estimated interval
            - level       :  level of pdf
        """
        nd = norm()
        self.dopoint(verbose=False)
        if kind == 'center':
            n = self.ymax / (nd.pdf(0) / (nd.pdf(nd.interval(conf)[0]) / 2))
            res = optimize.fsolve(self.func_twoside, n, args=(conf,), xtol=self.xtol, full_output=self.debug)
            self.interval = self.minmax(res[0])
        else:
            res = optimize.fsolve(self.func_oneside, self.point, args=(conf, kind), xtol=self.xtol, full_output=self.debug)
            if kind == 'right':
                self.interval = [res[0], self.x[-1]]
            elif kind == 'left':
                self.interval = [self.x[0], res[0]]

        if self.debug or verbose:
            print('interval:', self.interval[0], self.interval[1])
        return self.interval, res[0]


    def stats(self, conf=0.683, kind='center', latex=0, name=None):
        if name is not None:
            self.name = name
        self.dopoint(verbose=(latex<0))
        self.dointerval(conf=conf, kind=kind, verbose=(latex<0))
        if latex >= 0:
            print("{name} = {0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(self.point, self.interval[1] - self.point, self.point - self.interval[0], n=latex, name=self.name))

    def pdf(self, x):
        return self.inter(x)

    def rvs(self, n=1):
        class gen(rv_continuous):
            def __init__(self, parent, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent = parent

            def _pdf(self, x):
                return self.parent.pdf(x)

        return gen(self, a=self.x[0], b=self.x[-1]).rvs(size=n)

    def latex(self, f=2):
        return "{0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}".format(self.point, self.interval[1] - self.point, self.point - self.interval[0], n=f)


class distr2d():
    def __init__(self, x, y, z=None, num=None, xtol=1e-5, debug=False):
        if z is None:
            if num is None:
                num = int(np.sqrt(len(x)))
            self.x = np.linspace(np.min(x), np.max(x), num)
            self.y = np.linspace(np.min(y), np.max(y), num)
            X, Y = np.meshgrid(self.x, self.y)
            z = gaussian_kde([x, y])
            self.z = z(np.vstack([X.flatten(), Y.flatten()])).reshape(X.shape)

        elif len(z.shape) == 1 and len(x.shape) == 1 and len(y.shape) == 1:
            self.x = np.unique(sorted(x))
            self.y = np.unique(sorted(y))
            self.z = np.zeros([len(self.y), len(self.x)])
            for i, xi in enumerate(self.x):
                for k, yk in enumerate(self.y):
                    ind = np.where((xi == x) * (yk == y))[0]
                    if len(ind) > 0:
                        self.z[k, i] = z[ind[0]]
        elif len(x) in z.shape and len(y) in z.shape:
            self.x = x
            self.y = y
            self.z = z

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.zmax = None
        self.xtol = xtol
        self.debug = debug
        self.normalize()
        self.interpolate()
        #self.plot()

    def normalize(self):
        if 1:
            norm = np.abs(_simps(_simps(self.z, self.y, axis=0), self.x))
        else:
            # not working ???!!!
            inter = _interp2d(self.x, self.y, self.z, kind='linear', fill_value=0)
            norm = integrate.dblquad(inter, self.x[0], self.x[-1], lambda x: self.y[0], lambda x: self.y[-1])
        if self.debug:
            print('norm:', norm)
        self.z = self.z / norm

    def interpolate(self):
        if 1:
            self.inter = _interp2d(self.x, self.y, self.z, kind='cubic', fill_value=0)
            #self.inter = interpolate.Rbf(self.x, self.y, self.z, function='multiquadric', smooth=0.1)
        else:
            self.inter = interpolate.RectBivariateSpline(self.x, self.y, self.z, kx=3, ky=3)
        #xi, yi = 5, 14
        #print(self.inter(self.x[xi], self.y[yi]), self.z[yi, xi], self.x[xi], self.y[yi])
        #print(self.inter(self.x[xi+1], self.y[yi+1]), self.z[yi+1, xi+1], self.x[xi+1], self.y[yi+1])
        #print(self.inter((self.x[xi]+self.x[xi+1])/2, (self.y[yi]+self.y[yi+1])/2))

    def minter(self, x):
        return -self.inter(x[0], x[1])[0]

    def level(self, x, level):
        return np.squeeze(self.inter(x) - level)[()]  # scalar for root finders (NumPy >= 2)

    def minmax(self, level):
        if level > 0 and level < self.zmax:
            x = self.X[self.z > level].flatten()
            self.xmin, self.xmax = np.min(x), np.max(x)
            y = self.Y[self.z > level].flatten()
            self.ymin, self.ymax = np.min(y), np.max(y)
        elif level > self.zmax:
            self.xmin, self.xmax = self.point[0], self.point[0]
            self.ymin, self.ymax = self.point[1], self.point[1]
        elif level < 0:
            self.xmin, self.xmax = self.x[0], self.x[-1]
            self.ymin, self.ymax = self.y[0], self.y[-1]
        return [self.xmin, self.xmax], [self.ymin, self.ymax]

    def func(self, level, conf, x=None, y=None, z=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z
        zs = np.copy(z)
        zs[zs < level] = 0
        return _simps(_simps(zs, y, axis=0), x) - conf

    def dopoint(self, verbose=False):
        """
        Point estimate of distribution

        return: point, level
            - point         :  point estimate
        """
        ind = np.argwhere(self.z == np.nanmax(self.z.flatten()))
        self.point = optimize.fmin(self.minter, [self.x[ind[0][1]], self.y[ind[0][0]]], xtol=self.xtol, disp=self.debug)
        self.zmax = self.inter(self.point[0], self.point[1])
        if self.debug or verbose:
            print('point estimate:', self.point[0], self.point[1], self.zmax)
        return self.point

    def level(self, conf=0.683):
        """
        Level of pdf at given confidence level
        parameters:
            - conf           :  confidence level

        return: level
            - level          :  pdf value above pdf contains conf level of probability
        """
        x, y = np.linspace(np.min(self.x), np.max(self.x), 300), np.linspace(np.min(self.y), np.max(self.y), 300)
        z = self.inter(x, y)
        if 1:
            res = optimize.bisect(self.func, 0, np.squeeze(self.zmax)[()], args=(conf, x, y, z), xtol=self.xtol, disp=self.debug)
        else:
            res = optimize.fsolve(self.func, self.zmax/2, args=(conf, x, y, z), xtol=self.xtol, disp=self.debug)[0]
        if self.debug:
            print('fsolve:', res)

        return res

    def dointerval(self, conf=0.683, verbose=False):
        """
        Estimate the interval for the both parameters.
        parameters:
            - conf            :  confidence levels

        return: x_interval, y_interval, level
            - x_interval      :  interval values for x
            - y_interval      :  interval values for y
            - level           :  level of pdf corresponding to area equal confidence.
        """
        nd = norm()
        self.dopoint()
        n = self.zmax / (nd.pdf(0) / (nd.pdf(nd.interval(conf)[0]) / 2))
        res = optimize.fsolve(self.func, n, args=(conf,), xtol=self.xtol, full_output=self.debug)
        self.interval = self.minmax(res[0])

        level = self.level(conf=conf)
        self.interval = self.minmax(level)
        if self.debug or verbose:
            print('interval:', self.interval[0], self.interval[1])
        return self.interval[0], self.interval[1], res[0]

    def plot_contour(self, conf_levels=None, ax=None, xlabel='', ylabel='', limits=None, ls=['--'],
                     color='dodgerblue', color_point='gold', cmap='PuBu', alpha=1.0, colorbar=False,
                     font=14, title=None, zorder=1, label=None,lw= 0.5):
        if ax == None:
            fig, ax = plt.subplots()
        self.dopoint()
        if conf_levels == None:
            conf_levels = [0.683]
        if type(conf_levels) in [float, int]:
            conf_levels = [conf_levels]
        conf_levels = np.sort(conf_levels)[::-1]
        if cmap is not None:
            if isinstance(cmap, str):
                cmap = getattr(cm, cmap)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:,-1] = np.linspace(alpha*(0.6+0.4*alpha), 0.6+0.4*alpha, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            cs = ax.contourf(self.X, self.Y, self.z / self.zmax, 100, cmap=my_cmap, zorder=zorder,label = 'legend test')
        if color is not None:
            levels = [self.level(c) for c in conf_levels]
            print('levels',levels)
            if limits == None or limits == 0:
                c = ax.contour(self.X, self.Y, self.z / self.zmax, levels=levels / self.zmax, colors=color, lw=lw, linestyles='-',
                               zorder=zorder, label=label)
            else:
                c = ax.contour(self.X, self.Y, self.z / self.zmax, levels=levels / self.zmax, colors=color, lw=lw, linestyles='-', alpha=0)
                x, y = c.allsegs[0][0][:, 0], c.allsegs[0][0][:, 1]
                inter = interpolate.interp1d(x, y)
                x = np.linspace(x[0], x[-1], 30)
                ax.plot(x, inter(x), '-', c=color)
                if limits < 0:
                    lolims, uplims = False, True
                if limits > 0:
                    lolims, uplims = True, False
                x = np.linspace(x[0], x[-1], 10)
                ax.errorbar(x, inter(x), yerr=np.abs(limits), lolims=lolims, fmt='o', color=color, uplims=uplims, markersize=0,
                            capsize=0, zorder=zorder, label=label)
            if ls is not None:
                if hasattr(c, 'collections'):  # Matplotlib < 3.10
                    for c, s in zip(c.collections[:len(ls)], ls[::-1]):
                        c.set_dashes(s)
                else:
                    styles = list(c.get_linestyle())
                    for i, s in enumerate(ls[::-1][:len(styles)]):
                        styles[i] = s
                    c.set_linestyle(styles)
        if color_point is not None:
            ax.scatter(self.point[0], self.point[1], s=200, color=color_point, edgecolors='k', marker='*', zorder=50)
        if colorbar:
            fig.colorbar(cs, ax=ax) #, shrink=0.9)
        ax.set_xlabel(xlabel, fontsize=font)
        ax.set_ylabel(ylabel, fontsize=font)
        ax.tick_params(axis='both', which='major', labelsize=font)
        if title is not None:
            ax.set_title(title, fontdict={'fontsize': font})
        return ax

    def plot(self, fig=None, frac=0.1, indent=0.15, color_marg='dodgerblue',
             conf_levels=None, xlabel='', ylabel='', limits=None, ls=None,
             color='greenyellow', color_point='gold', cmap='PuBu', alpha=1.0, colorbar=False,
             font=14, title=None, zorder=1):
        if fig is None:
            fig = plt.figure()
        ax = fig.add_axes([indent, indent, 1 - indent - frac, 1 - indent - frac])
        ax.set_position([indent, indent, 1 - indent - frac, 1 - indent - frac])
        self.plot_contour(ax=ax, conf_levels=conf_levels, xlabel=xlabel, ylabel=ylabel, limits=limits, ls=ls,
                     color=color, color_point=color_point, cmap=cmap, alpha=alpha, colorbar=colorbar,
                     font=font, title=title, zorder=zorder)

        x, y = ax.get_xlim(), ax.get_ylim()
        #helper = floating_axes.GridHelperCurveLinear(Affine2D().rotate_deg(-90), extremes=(0, 4, 0, 4))
        #ax = floating_axes.FloatingSubplot(fig, [1 - frac, indent, frac, 1 - indent - frac], grid_helper=helper)

        dy = self.marginalize('x')
        ax = fig.add_axes([1 - frac, indent, frac, 1 - indent - frac])
        #ax.set_position([1 - frac, indent, frac, 1 - indent - frac])
        ax.set_axis_off()
        ax.plot(dy.inter(dy.x), dy.x, '-', color=color_marg, lw=1.5)
        ax.set_ylim(y)
        ax.set_xlim([0 + ax.get_xlim()[1]/50, ax.get_xlim()[1]])

        dx = self.marginalize('y')
        ax = fig.add_axes([indent, 1 - frac, 1 - indent - frac, frac])
        #ax.set_position([indent, 1 - frac, 1 - indent - frac, frac])
        ax.set_axis_off()
        ax.plot(dx.x, dx.inter(dx.x), '-', color=color_marg, lw=1.5)
        ax.set_xlim(x)
        ax.set_ylim([0 + ax.get_ylim()[1]/50, ax.get_ylim()[1]])

        fig.tight_layout()
        return fig

    def marginalize(self, over='y'):
        """
        marginaize to 1d distribution over specified axis
        parameters:
            - over              :  axis over which perform marginalization, 'x' or 'y'
        return: distr
            - distr             :  1d distribution object
        """
        if over == 'y':
            return distr1d(self.x, _simps(self.z, self.y, axis=0), debug=self.debug)
        else:
            return distr1d(self.y, _simps(self.z, self.x, axis=1), debug=self.debug)

    def pdf(self, x, y):
        """
        return pdf of the distribution
        parameters:
            - x          :  x
            - y          :  y
        """
        return self.inter(x, y)

    def rvs(self, n=1, xrange=None, yrange=None):
        """
        Generate the random sample from the distribution using rejection sampling
        parameters:
            - n            : sample size to generate
            - xrange       : range of x to sample
            - yrange       : range of y to sample
        return: sample,
            - sample       : sample
        """
        if self.zmax is None:
            self.dopoint()
        print(self.zmax)

        if xrange is None:
            xrange = (self.x[0], self.x[-1])

        if yrange is None:
            yrange = (self.y[0], self.y[-1])

        x, y = [-2, -1], [-1.5, -0.5]
        num = n
        genx, geny = [], []
        while len(genx) < n:
            x = np.random.uniform(xrange[0], xrange[1], num)
            y = np.random.uniform(yrange[0], yrange[1], num)
            w = np.random.uniform(size=num)
            pdf = np.array([self.pdf(xx, yy)[0] for xx, yy in zip(x, y)])
            #pdf = np.diag(self.pdf(x, y))
            mask = (pdf / self.zmax) > w
            #if mask.sum() > 0:
            #    print(pdf[mask] / self.zmax, w[mask], x[mask], y[mask])
            #print(genx, geny)
            genx += list(x[mask])
            geny += list(y[mask])
        m = np.random.choice(len(genx), n)
        return np.asarray(genx)[m], np.asarray(geny)[m]


# =====================================================================================
# species / column density container  (from spectro/atomic.py)
# =====================================================================================

class e():
    """
    Class for species: 
    
        purpose: set observational data for species 
        
        dependences:    a_unc 
                        mendeleev
                        re
                        
        
        to initialize species:
            1. e('SiII', 19.0, 0.2, 0.2)
            2. e('SiII', a(19.0, 0.2, 0.2))
            3. e('SiII', '19.0\pm0.2')
            4. e('SiII', '19.0^{+0.2}_{-0.2}')
            
            where 19.0, 0.2, 0.2 sets column density of species.
            column density will be saved as <a> object in self.col value.
            
        to initialize upper/lower limit:
            1. e('SiII*', '<19')
            2. e('SiII*', a(19, t='u')),  see also description of <a> object
        
        Molecular species (H2, HD, CO, ...) can be specified as well:
            1. e('H2', 21.0, 0.1, 0.1, J=0)
            2. e('H2', 14.0, 0.1, 0.1, J=0, nu=1)
            
            nu=0 by default
            
        you can also specify b parameter for species, using <b> keyword
            1. e('H2', 18.0, 0.1, 0.1, J=4, b=(2.4, 0.4, 0.8))
            
        notes:
        1. ionization states gets automatically by get_ioniz() method
        2. a list of method are provided coupled with mendeleev package.
        
    """
    def __init__(self, *args, **kwargs):
        
        # representation name SiII, OI*, HD
        self.name = args[0]
        # short desciprion of element, like Si
        self.el = self.get_element_name()
        
        # set column density:
        if len(args) == 1:
            self.col = a()
        elif len(args) == 2:
            if isinstance(args[1], str):
                self.col = a(args[1])
            elif isinstance(args[1], a):
                self.col = args[1]
            elif isinstance(args[1], (int, float)):
                self.col = a(args[1], 0, 0)
        elif len(args) == 3:
            self.col = a(args[1], args[2], args[2])
        elif len(args) == 4:
            self.col = a(args[1], args[2], args[3])
        else:
            self.col = a(args[1], args[2], args[3], )
        if 'f' in kwargs.keys():
            if kwargs['f'] in ['d', 'dec']:
                self.col.repr = 'dec'
                self.col.default_format = 'dec'
            if kwargs['f'] in ['l', 'log']:
                self.col.repr = 'log'
                self.col.default_format = 'log'

        self.nu = None
        self.J = None

        name = self.name[:]
        if 'v' in name and name[name.index('v')+1:].isnumeric():
            self.nu = int(name[name.index('v')+1:])
            name = name[:int(name.index('v'))]
        if 'j' in name:
            self.J = int(name[int(name.index('j'))+1:])
        self.energy = None
        self.b = None
        
        for k in kwargs.items():
            if k[0] == 'b' and isinstance(k[1], (list, tuple)):
                setattr(self, k[0], a(k[1][0], k[1][1], k[1][2], 'd'))
            else:
                setattr(self, k[0], k[1])
        
        # refresh representation for molecules
        self.name = self.__repr__()
        
        self.mend = None
        
        self.fine = self.name.count('*')
        
        self.statw()
        self.ionstate()

        self.lines = []

        # for line modeling:
        self.logN = self.col.val
        if self.b is not None:
            self.b_ = self.b.val
        
        # some additional parameters:        
        self.mod = 0
        
    def get_element_name(self):
        st = self.name[:]
        if 'j' in st:
            st = st[:st.index('j')]
        for s in ['I', 'V', 'X', '*']:
            st = st.replace(s, '')
            
        return st
        
    def __eq__(self, other):
        return self.name == other.name and self.ion == other.ion

    def __repr__(self):
        st = self.name
        if self.name in ['H2', 'HD', 'CO']:
            if self.J is not None:
                st +=  'j' + str(self.J)
            if self.nu is not None:
                st += 'v' + str(self.nu)
        return st
        
    def __str__(self):
        return self.__repr__() + ' ' + str(self.col)
    
    def info(self):
        """
        print info about element
        """
        seq = ['name', 'el', 'ioniz', 'fine', 'J', 'nu', 'col']
        attrs = vars(self).copy()
        for s in seq:
            print('{:}: {:}'.format(s, attrs[s]))
            del attrs[s]
        for a in attrs.items():
            print('{:}: {:}'.format(a[0], a[1]))
        
    def statw(self):
        """
        return statistical weight of the element
        """
        self.stat = None
        if 'H2' in self.name:
            if self.J is not None:
                self.stat = (2 * self.J + 1) * ((self.J % 2) * 2 + 1)

        elif 'HD' in self.name:
            if self.J is not None:
                self.stat = (2 * self.J + 1)
        
        elif 'CO' in self.name:
            if self.J is not None:
                self.stat = (2 * self.J + 1)

        elif 'SiII' in self.name:
            self.stat = (self.fine + 1) * 2
            
        elif 'CII' in self.name:
            self.stat = (self.fine + 1) * 2
        
        elif 'CI' in self.name:
            self.stat = self.fine * 2 + 1
        
        elif 'OI' in self.name:
            self.stat = self.fine * 2 + 1
        
        return self.stat
        
    def ionstate(self):
        """
        Method to set ionization state by name
        """
        self.ioniz = 0
        m = re.search(r"[IXV]", self.name)
        if m:
            string = self.name[m.start():]
            roman = [['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]]
            for letter, value in roman:
                while string.startswith(letter):
                    self.ioniz += value
                    string = string[len(letter):]
                    
        return self.ioniz
    
    def mendeleev(self, elem=None):
        if elem is None:
            elem = self.el
        if self.mend is None:
            from mendeleev import element  # optional dependency, only needed here
            self.mend = element(elem)

    def atom(self):
        self.mendeleev()        
        return self.mend.atomic_number
    
    def ion_energy(self):
        self.mendeleev()
        return self.mend.ionenergies[self.ioniz]
        
    def mass(self):
        if 'D' not in self.name:
            self.mendeleev()
            return self.mend.mass
        else:
            self.mendeleev('H')
            return self.mend.mass * 2

    def cond_temp(self):
        return elem.CondensTemperature(self.name+'_')
        
    @classmethod
    def read_line(cls, descr):
        """
        Method to read species data from the line
        Used by reading data from file
        """
        s = cls()
        s.readline(descr)
        return s

    def readline(self, descr):
        """
        Method to read species data from the line
        Used by reading data from file
        """
        words = descr.split()
        self.descr = descr

        self.name = words[0]
        i = 1

        if self.name in ['H2', 'HD', 'CO']:

            if len(words) == 4:
                self.J = -1
                self.nu = -1
            else:
                if len(words) == 6:
                    self.nu = int(words[i])
                    i += 1
                elif len(words) == 5:
                    self.nu = 0
                self.J = int(words[i])
                self.stat = (2*self.J+1)*(2*(self.J%2)+1)
                i += 1

        if self.name in ['OI', 'CI', 'SiII']:
            if len(words) == 5:
                self.J = int(words[1])
                i += 1
            else:
                self.J = -1

        self.get_ioniz()
        self.col = a(float(words[i]), float(words[i+1]), float(words[i+2]))

# =====================================================================================
# loading old pickles
# =====================================================================================
_RENAMED_MODULES = {
    'spectro.a_unc': __name__, 'spectro.stats': __name__, 'spectro.atomic': __name__,
    'a_unc': __name__, 'stats': __name__, 'atomic': __name__,
    'spectro.pyratio': 'pyratio',
    'H2_exc': 'PDR_exc', 'H2_excitation.H2_exc': 'PDR_exc',
    'H2_summary': 'PDR_systems', 'H2_excitation.H2_summary': 'PDR_systems', 'PDR_summary': 'PDR_systems',
    'H2viewer': 'PDRviewer',
}


_RENAMED_CLASSES = {'H2_exc': 'PDR_exc', 'H2viewer': 'PDRviewer'}


class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = _RENAMED_MODULES.get(module, module)
        if module in ('PDR_exc', 'PDRviewer'):
            name = _RENAMED_CLASSES.get(name, name)
        return super().find_class(module, name)


def load_pickle(filename):
    """pickle.load() that also reads files saved with spectro or the old H2_* module names"""
    with open(filename, 'rb') as f:
        return _RenamingUnpickler(f).load()
