#definition theano.scalar op that have their python implementation taked from scipy
#as scipy is not always available, we put threat them separatly
import numpy

from theano.scalar.basic import (UnaryScalarOp,
                                 exp, upgrade_to_float,
                                 float_types)
from theano.scalar.basic import (upgrade_to_float_no_complex,
                                 complex_types,
                                 upcast)

imported_scipy_special = False
try:
    import scipy.special
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass


class Erf(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erf(x)
        else:
            super(Erf, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return gz * cst * exp(-x * x),
        else:
            return None,

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erf(%(x)s);" % locals()
erf = Erf(upgrade_to_float, name='erf')


class Erfc(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfc(x)
        else:
            super(Erfc, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return - gz * cst * exp(-x * x),
        else:
            return None,

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name='erfc')


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Note: This op can still be executed on GPU, despite not having c_code.  When
    running on GPU, sandbox.cuda.opt.local_gpu_elemwise_[0,1] replaces this op
    with sandbox.cuda.elemwise.ErfinvGPU.

    (TODO) Find a C implementation of erfinv for CPU.
    """
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfinv(x)
        else:
            super(Erfinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return gz * cst * exp(erfinv(x) ** 2),
        else:
            return None,

    def c_support_code(self):
        #taken from: https://github.com/stevengj/julia/commit/9c24795ac2918df7b8cba9bb129db8c535d321e8
        return """
#ifndef _PSIFUNCDEFINED
#define _ERFINVFUNCDEFINED<
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif 

// Compute the inverse of the error function: erf(erfinv(x)) == x, 
// using the rational approximants tabulated in:
//     J. M. Blair, C. A. Edwards, and J. H. Johnson, "Rational Chebyshev 
//     approximations for the inverse of the error function," Math. Comp. 30,
//     pp. 827--830 (1976).
//         http://dx.doi.org/10.1090/S0025-5718-1976-0421040-7 
//         http://www.jstor.org/stable/2005402
DEVICE double erfinv(double x){
    double a = abs(x);
    if (a >= 1.0){
        if (x == 1.0)
            return INFINITY; //inf(Float64);
        else if (x == -1.0)
            return -INFINITY; //-inf(Float64);
        //TODO
        //throw(DomainError())
    }else if( a <= 0.75){ // Table 17 in Blair et al.
        double t = x*x - 0.5625;
// evaluate p[1] + x * (p[2] + x * (....)), i.e. a polynomial via Horner's rule
/*(macro horner(x, p...)
    ex = p[end]
    for i = length(p)-1:-1:1
        ex = :($(p[i]) + $x * $ex)
    end
    ex
end*/
        double tab1[] = {0.160304955844066229311e2, 
                             -0.90784959262960326650e2,
                              0.18644914861620987391e3,
                             -0.16900142734642382420e3,
                              0.6545466284794487048e2,
                             -0.864213011587247794e1,
                              0.1760587821390590};
        double tab2[] = {0.147806470715138316110e2,
                             -0.91374167024260313936e2,
                              0.21015790486205317714e3,
                             -0.22210254121855132366e3,
                              0.10760453916055123830e3,
                             -0.206010730328265443e2,
                              0.1e1};
        double v1 = tab1[6];
        for(int i=5; i>=0; i--)
            v1 = tab1[i] + t * v1;
        double v2 = tab2[6];
        for(int i=5; i>=0; i--)
            v2 = tab2[i] + t * v2;
        return x * v1 / v2;
    }else if (a <= 0.9375){ // Table 37 in Blair et al.
        double t = x*x - 0.87890625;
            double tab1[] = {-0.152389263440726128e-1,
                               0.3444556924136125216,
                              -0.29344398672542478687e1,
                               0.11763505705217827302e2,
                              -0.22655292823101104193e2,
                               0.19121334396580330163e2,
                              -0.5478927619598318769e1,
                               0.237516689024448};
            double tab2[] = {-0.108465169602059954e-1,
                               0.2610628885843078511,
                              -0.24068318104393757995e1,
                               0.10695129973387014469e2,
                              -0.23716715521596581025e2,
                               0.24640158943917284883e2,
                              -0.10014376349783070835e2,
                               0.1e1};
        double v1 = tab1[7];
        for(int i=6; i>=0; i--)
            v1 = tab1[i] + t * v1;
        double v2 = tab2[7];
        for(int i=6; i>=0; i--)
            v2 = tab2[i] + t * v2;
        return x * v1 / v2;
    }else{ // Table 57 in Blair et al.
        double t = 1.0 / sqrt(-log(1.0 - a));
        double tab1[] = {0.10501311523733438116e-3,
                          0.1053261131423333816425e-1,
                          0.26987802736243283544516,
                          0.23268695788919690806414e1,
                          0.71678547949107996810001e1,
                          0.85475611822167827825185e1,
                          0.68738088073543839802913e1,
                          0.3627002483095870893002e1,
                          0.886062739296515468149};
        double tab2[] = {0.10501266687030337690e-3,
                          0.1053286230093332753111e-1,
                          0.27019862373751554845553,
                          0.23501436397970253259123e1,
                          0.76078028785801277064351e1,
                          0.111815861040569078273451e2,
                          0.119487879184353966678438e2,
                          0.81922409747269907893913e1,
                          0.4099387907636801536145e1,
                          0.1e1};
        double v1 = tab1[8];
        for(int i=7; i>=0; i--)
            v1 = tab1[i] + t * v1;
        double v2 = tab2[8];
        for(int i=7; i>=0; i--)
            v2 = tab2[i] + t * v2;
        return v1 / (copysign(t, x) * v2);
    }
} 
#endif
"""
    # TODO: erfinv() is not provided by the C standard library
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfinv(%(x)s);" % locals()

erfinv = Erfinv(upgrade_to_float_no_complex, name='erfinv')


class Erfcinv(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcinv(x)
        else:
            super(Erfcinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return - gz * cst * exp(erfcinv(x) ** 2),
        else:
            return None,

    # TODO: erfcinv() is not provided by the C standard library
    def _c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        """
double erfcinv(double y){
    if (y > 0.0625)
        return erfinv(1.0 - y);
    else if (y <= 0.0){
        if (y == 0.0)
            return INFINITY; //inf(Float64)
        // TODO
        throw(DomainError())
    } else if (y >= 1e-100){ # Table 57 in Blair et al.
        double t = 1.0 / sqrt(-log(y));
        return @horner(t, 0.10501_31152_37334_38116e-3,
                          0.10532_61131_42333_38164_25e-1,
                          0.26987_80273_62432_83544_516,
                          0.23268_69578_89196_90806_414e1,
                          0.71678_54794_91079_96810_001e1,
                          0.85475_61182_21678_27825_185e1,
                          0.68738_08807_35438_39802_913e1,
                          0.36270_02483_09587_08930_02e1,
                          0.88606_27392_96515_46814_9) / 
              (t *
               @horner(t, 0.10501_26668_70303_37690e-3,
                          0.10532_86230_09333_27531_11e-1,
                          0.27019_86237_37515_54845_553,
                          0.23501_43639_79702_53259_123e1,
                          0.76078_02878_58012_77064_351e1,
                          0.11181_58610_40569_07827_3451e2,
                          0.11948_78791_84353_96667_8438e2,
                          0.81922_40974_72699_07893_913e1,
                          0.40993_87907_63680_15361_45e1,
                          0.1e1))
    } else { # Table 80 in Blair et al.
        double t = 1.0 / sqrt(-log(y));
        return @horner(t, 0.34654_29858_80863_50177e-9,
                          0.25084_67920_24075_70520_55e-6,
                          0.47378_13196_37286_02986_534e-4,
                          0.31312_60375_97786_96408_3388e-2,
                          0.77948_76454_41435_36994_854e-1,
                          0.70045_68123_35816_43868_271e0,
                          0.18710_42034_21679_31668_683e1,
                          0.71452_54774_31351_45428_3e0) /
          (t * @horner(t, 0.34654_29567_31595_11156e-9,
                          0.25084_69079_75880_27114_87e-6,
                          0.47379_53129_59749_13536_339e-4,
                          0.31320_63536_46177_68848_0813e-2,
                          0.78073_48906_27648_97214_733e-1,
                          0.70715_04479_95337_58619_993e0,
                          0.19998_51543_49112_15105_214e1,
                          0.15072_90269_27316_80008_56e1,
                          0.1e1))
    }
}
        """
        return "%(z)s = erfcinv(%(x)s);" % locals()

erfcinv = Erfcinv(upgrade_to_float_no_complex, name='erfcinv')


class Gamma(UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        return scipy.special.gamma(x)

    def impl(self, x):
        return Gamma.st_impl(x)

    def grad(self, (x, ), (gz, )):
        return gz * gamma(x) * psi(x),

    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in float_types:
            return """%(z)s = tgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
gamma = Gamma(upgrade_to_float, name='gamma')


class GammaLn(UnaryScalarOp):
    """
    Log gamma function.
    """
    @staticmethod
    def st_impl(x):
        return scipy.special.gammaln(x)

    def impl(self, x):
        return GammaLn.st_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                lgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
gammaln = GammaLn(upgrade_to_float, name='gammaln')


class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.
    """
    @staticmethod
    def st_impl(x):
        return scipy.special.psi(x)

    def impl(self, x):
        return Psi.st_impl(x)

    def grad(self, inputs, outputs_gradients):
        raise NotImplementedError()
        return [None]

    def c_support_code(self):
        return (
"""
// For GPU support
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef _PSIFUNCDEFINED
#define _PSIFUNCDEFINED
DEVICE double _psi(double x){

    /*taken from
    Bernardo, J. M. (1976). Algorithm AS 103:
    Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
    http://www.uv.es/~bernardo/1976AppStatist.pdf */

    double y, R, psi_ = 0;
    double S  = 1.0e-5;
    double C = 8.5;
    double S3 = 8.333333333e-2;
    double S4 = 8.333333333e-3;
    double S5 = 3.968253968e-3;
    double D1 = -0.5772156649;

    y = x;

    if (y <= 0.0)
        return psi_;

    if (y <= S )
        return D1 - 1.0/y;

    while (y < C){
        psi_ = psi_ - 1.0 / y;
        y = y + 1;}

    R = 1.0 / y;
    psi_ = psi_ + log(y) - .5 * R ;
    R= R*R;
    psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

    return psi_;}
    #endif
        """ )

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _psi(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
psi = Psi(upgrade_to_float, name='psi')
