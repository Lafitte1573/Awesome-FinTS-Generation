# Regression estimation for continuous-time functional data processes with missing at random response

Mohamed Chaouch & Naâmane Laïb

To cite this article: Mohamed Chaouch & Naâmane Laïb (22 Mar 2024): Regression estimation for continuous-time functional data processes with missing at random response, Journal of Nonparametric Statistics, DOI: 10.1080/10485252.2024.2332686

To link to this article: https://doi.org/10.1080/10485252.2024.2332686

# Regression estimation for continuous-time functional data processes with missing at random response

Mohamed Chaoucha and Naâmane Laïbb aProgram of Statistics, Department of Mathematics, Statistics and Physics, College of Arts and Sciences, Qatar University, Doha, Qatar; bLaboratoire AGM, UMR 8088-CNRS, CY Cergy Paris Univeristé, Cergy, France

# ABSTRACT

In this paper, we are interested in nonparametric kernel estimation of a generalised regression function based on an incomplete sample $( X _ { t } , Y _ { t } , \zeta _ { t } ) _ { t \in [ 0 , T ] }$ copies of a continuous-time stationary and ergodic process $( X , Y , \zeta )$ . The predictor $X$ is valued in some infinitedimensional space, whereas the real-valued process Y is observed when the Bernoulli process $\zeta = 1$ and missing whenever $\zeta = 0$ . Uniform almost sure consistency rate as well as the evaluation of the conditional bias and asymptotic mean square error are established. The asymptotic distribution of the estimator is provided with a discussion on its use in building asymptotic confidence intervals. To illustrate the performance of the proposed estimator, a first simulation is performed to compare the efficiency of discrete-time and continuoustime estimators. A second simulation is conducted to discuss the selection of the optimal sampling mesh in the continuous-time case. Then, a third simulation is considered to build asymptotic confidence intervals. An application to financial time series is used to study the performance of the proposed estimator in terms of point and interval prediction of the IBM asset price log-returns. Finally, a second application is introduced to discuss the usage of the initial estimator to impute missing household-level peak electricity demand.

# ARTICLE HISTORY

Received 29 April 2021   
Accepted 4 March 2024

# KEYWORDS

Asymptotic mean square error; continuous-time ergodic processes; confidence intervals; functional data; generalised regression; missing at random

# 1. Introduction

Let $( \mathcal { E } , d )$ be an infinite-dimensional space equipped with a semi-metric $d ( \cdot , \cdot )$ . Consider $( X _ { t } , Y _ { t } , \zeta _ { t } ) _ { t \in \mathbb { R } ^ { + } }$ a stationary and ergodic continuous-time process valued in the space $\mathcal { X } : = \mathcal { E } \times \mathbb { R } \times \{ 0 , 1 \}$ , such that each triplet $( X _ { t } , Y _ { t } , \zeta _ { t } )$ has the same probability distribution as the random variable (r.v.) $( X , Y , \zeta )$ defined on probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ . Let S be a compact interval in $\mathbb { R }$ and $\psi$ be a measurable function defined on the space $S \times \mathbb { R }$ , $( y , Y ) \mapsto \psi ( y , Y )$ , where $y$ is a real variable such that $\mathbb { E } ( | \psi ( Y , y ) | ) < \infty$ . Consider the following regression model:

$$
\psi ( Y , y ) = m _ { \psi } ( X , y ) + \varepsilon ,
$$

where $m _ { \psi } ( X , y )$ is the conditional expectation of $\psi ( Y , y )$ given the r.v. $X$ . That is, for any $y \in S$ and a fixed $x \in { \mathcal { E } }$ $\in \mathcal { E } , \mathbb { E } ( \psi ( Y , y ) | X = x ) = m _ { \psi } ( x , y ) ,$ . The error term $\varepsilon$ is independent of $X$ such $\mathbb { E } ( \varepsilon \mid X ) = 0$ almost surely (a.s.)

Usually, when no data are missing, a sample of stationary and ergodic process $( X _ { t } , Y _ { t } ) _ { 0 \leq t \leq T }$ is observed. Here, we allow the response variable $Y _ { t }$ to be Missing At Random (MAR) any time t. To check whether an observation is complete or missing, a new variable $\zeta$ is introduced into the model as an indicator of the missing observations. Thus, for any $t \in [ 0 , T ]$ , $\zeta _ { t } = 1$ if $Y _ { t }$ is observed and 0 if $Y _ { t }$ is missing. We suppose that the Bernoulli random variable $\zeta$ satisfies $\mathbb { P } ( \zeta = 1 | X = x , Y = y ) = \mathbb { P } ( \zeta = 1 | X = x ) : = p ( x )$ . Here, $p ( x ) > 0$ is the conditional probability of observing the response variable and is usually unknown. This assumption allows to conclude that $\zeta$ and $Y$ are conditionally independent given $X$ . Note that the above assumption says that the response variable does not provide additional information, on top of that given by the explanatory variable, to predict whether an individual will present a missing response.

In this paper, we are interested in the estimation of the regression function $m _ { \psi } ( \cdot , y )$ based on the observed data $( X _ { t } , Y _ { t } , \zeta _ { t } ) _ { 0 \leq t \leq T } .$ Note that, for any $t \in [ 0 , T ] , t \mapsto \{ X _ { t } ( \omega ) , \omega \in$ $\Omega \}$ is an element in the space $\mathcal { E }$ , which means that, for any fixed time $t = t _ { 0 } , X _ { t _ { 0 } }$ is a curve. Specifically, if ${ \mathcal { E } = \mathcal { C } _ { [ 0 , 1 ] } }$ is the space of square integrable functions defined on [0, 1], then the predictor $X _ { t _ { 0 } } : = \{ X _ { t _ { 0 } } ( s ) : s \in [ 0 , 1 ] \}$ describes a trajectory in the functional space $\mathcal { E }$ observed at the fixed time $t _ { 0 }$ .

In real life, there are several situations where the response variable might be missing at random. For instance, in survey sampling studies the non-response is an increasingly common problem, where the missing response reaches rates of $2 5 \% { - } 3 0 \%$ or even higher (see, e.g. Sikov 2018). In such cases, the missing data become a real source of bias in survey sampling estimation. Another case where the response may be subject to the MAR phenomena is the household electricity consumption monitoring. Indeed, the real-time collection of intra-day electricity consumption is now possible after the deployment of smart meters at the household level. The transmission of the information from the smart meter towards the information system goes usually through WIFI or optical fibre networks which are significantly dependent on the weather conditions, among other factors. Therefore, a response variable such as the daily total electricity consumption might be subject to missing at random mechanism due to bad weather conditions (for more details , see Section 5.2). In financial market, despite the modern technology, which allows to collect data at a very fine time scale, financial data can still be missing. For instance, there are some regular holidays, such as Thanksgiving Day and Christmas, for which stock price data are missing. There are many other technical reasons (such as breakdown in devises recording data, computers’ sudden shutdowns, . . . ) that make stretches of data missing (see Section 5.1 for more details about this application). For further examples and details about missing at random data the reader is referred to Chapter 1 in Little and Rubin (2002).

Whenever $( X _ { i } , Y _ { i } , \zeta _ { i } ) _ { 1 \leq i \leq n }$ is an independent and identically distributed (i.i.d) random sample several authors investigated nonparametric and semiparametric estimations of the regression function. In the framework where $X$ is a finite dimensional, one can quote (Cheng 1994; Little and Rubin 2002; Nittner 2003; Tsiatis 2006; Liang et al. 2007; Efromovich 2011). See also, Ferraty et al. (2013) when the predictor is infinite-dimensional. However, less attention has been given to the case of dependent data including an infinite-dimensional covariate, except Ling et al. (2015), where a local constant estimation of the regression operator with discrete-time ergodic processes was considered.

In the case of continuous-time finite dimensional processes $( X , Y ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d ^ { \prime } }$ ( $d$ and $d ^ { \prime } \geq 1$ ) satisfying a strong mixing condition, the estimation of the regression function based on completely observed data was considered by several authors, see for instance, the monograph by Bosq (1998) and the references therein.

Some of these results were extended by Didi and Louani (2014) and Bouzebda and Didi (2017) for stationary and ergodic processes. Chaouch and Laïb (2019) studied the asymptotic mean square error of the kernel regression estimator for MAR stationary and ergodic process and obtained an explicit upper bound of it.

It is worth noting that, even though a continuous-time functional processes framework is considered in this paper, in practice data are often collected according to some sampling scheme and the continuous-time process is discretised. Our results are then valid when considering a discrete-time ergodic stationary process $( X _ { t _ { k } } , Y _ { t _ { k } } , \zeta _ { t _ { k } } ) _ { 1 \leq k \leq n }$ sampled from a continuous-time process $\{ ( X _ { t } , Y _ { t } , \zeta _ { t } ) \} _ { 0 \leq t \leq T }$ with a regular sampling mesh $\delta = T / n$ . A discussion on the optimal choice of the sampling mesh in the continuous-time case will be then of great practical interest (see Section 3.5 and Simulation 2).

In the setting where $( X _ { t } , Y _ { t } ) _ { t \in [ 0 , T ] } \in \mathcal { E } \times \mathbb { R }$ is an $\alpha$ -mixing continuous-time process with $Y$ is completely observed, Maillot (2008) established the convergence with rates of the regression operator, and a super-optimal mean square convergence rate was obtained in Chesneau and Maillot (2014).

This paper aims to complete and extend (Maillot 2008; Chesneau and Maillot 2014; Ling et al. 2015) work at several levels. First, we suppose that the continuous-time process satisfies an ergodic assumption rather than an $\alpha$ -mixing one. Therefore, the dependence structure considered here is more general and involves several processes which do not satisfy the mixing property. Indeed, our results are valid for $\alpha$ -mixing and non $\alpha$ -mixing as well as long memory and Bernoulli shift processes (for more details, see examples used to discuss Assumption A3 below). Moreover, our results are stated and proved without assuming neither mixing condition nor imposing a particular covariance structure on the process. This is due to the fact that the main technical tools used here are martingale difference devices and sequence of projections on appropriate $\sigma$ -fields. Second, we complete and extend results established in Ling et al. (2015) for discrete-time functional data processes, to the continuous-time functional framework.

We estimate a general operator $m _ { \psi } ( x , y )$ which includes conditional mean, conditional distribution function and conditional quantiles. It is worth noting that such extension is not obvious since it requires an appropriate definition of $\sigma$ -fields adapted to continuoustime context. Such adaptation is crucial when using martingale difference tools to establish asymptotic properties of the estimator. Third, the response variable considered here is affected by the MAR mechanism and therefore is not completely observed as in Maillot (2008). Moreover, in contrast to Maillot (2008) and Chesneau and Maillot (2014), we do not limit our study to the mean square convergence, but we provide a more exhaustive inference on the regression operator estimator including pointwise and uniform almost sure convergence rate, identification of the limiting distribution of our estimator, and provide method to build confidence intervals. Fourth, simulation study is also carried out to investigate the selection of the ‘optimal’ sampling mesh, which is one of the most important topics in nonparametric estimation with continuous-time processes.

The rest of this paper is organised as follows. In Section 2, we present the framework adapted to continuous-time ergodic processes and introduce assumptions needed for establishing asymptotic results. The main asymptotic properties of the estimator are discussed in Section 3. An illustration of the performance of the proposed estimator is discussed through simulated data in Section 4. Section 5 is devoted to an application of the proposed estimator to financial time series. Section 6 discusses the application of our theoretical results to continuous-time conditional quantiles estimation. Finally technical proofs are given in the Appendix.

# 2. Framework and assumptions

To define the framework of our study, we need to introduce some definitions. Let $X =$ $( X _ { t } ) _ { t \in [ 0 , \infty ) }$ be a continuous-time process defined on a probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ and observed at any time $t \in [ 0 , T ]$ . For more details about the definition of continuous-time ergodic processes, the reader is referred to Didi and Louani (2014). From now on, we consider $\{ \mathcal { F } _ { t } , t \geq 0 \}$ the filtration defined on $( \Omega , { \mathcal { F } } )$ , that is $\{ \mathcal { F } _ { t } , t \geq 0 \}$ is an increasing sequence of sub- $\sigma$ -algebras of $\mathcal { F }$ .

For a positive real number $\delta$ such that $\begin{array} { r } { n = \frac { T } { \delta } \in \mathbb { N } } \end{array}$ and $j \in \mathbb { N } \cap [ 1 , n ]$ , consider the $\delta$ - partition $( T _ { j } = j \delta ) _ { 1 \leq j \leq n }$ of the interval $[ 0 , T ]$ . Furthermore, for $t > 0$ and $1 \leq j \leq n$ , we define the following $\sigma$ -fields:

$$
\begin{array} { r l } & { \mathcal { F } _ { t - \delta } : = \sigma ( ( X _ { s } , Y _ { s } ) : 0 \le s < t - \delta ) , \quad \mathcal { F } _ { j } : = \mathcal { F } _ { T _ { j } } = \sigma ( ( X _ { s } , Y _ { s } ) , 0 \le s < T _ { j } ) , } \\ & { \quad S _ { t , \delta } : = \sigma ( ( X _ { s } , Y _ { s } ) , \quad ( X _ { r } ) : 0 \le s < t , t \le r \le t + \delta ) . } \end{array}
$$

If $t < 0$ we take $\mathcal { F } _ { t }$ the trivial $\sigma$ -field. Note that, for any $\delta > 0$ and $t > 0$ , we have $\mathcal { F } _ { t - \delta } \subset$ $S _ { t - \delta , \delta } \subset S _ { t , \delta }$ . Moreover, for any $j \ge 2$ , such that $T _ { j - 1 } \leq t \leq T _ { j } ,$ , we have $\mathcal { F } _ { j - 2 } \subseteq \mathcal { F } _ { t - \delta } \subset$ $S _ { t , \delta }$ .

Let $B ( x , u )$ be a ball centred at $x \in { \mathcal { E } }$ with radius $u > 0$ . Denote $D _ { t } : = d ( x , X _ { t } )$ a nonnegative real-valued continuous-time process and let $F _ { x } ( u ) = \mathbb { P } ( D _ { t } \leq u ) : = \mathbb { P } ( X _ { t } \in$ $B ( x , u ) )$ and $F _ { x } ^ { \mathcal { F } _ { t - \delta } } = \mathbb { P } ( X _ { t } \in B ( x , u ) | \mathcal { F } _ { t - \delta } )$ be the distribution function and conditional distribution function of $( D _ { t } ) _ { 0 \leq t \leq T }$ given the $\sigma$ -field $\mathcal { F } _ { t - \delta }$ , respectively.

To define an estimator of the regression function adapted to the MAR, multiply Equation (1) by $\zeta$ to get

$$
\zeta \psi ( Y , y ) = \zeta m _ { \psi } ( X , y ) + \zeta \varepsilon .
$$

Taking conditional expectations with respect to $X = x ,$ one gets $\begin{array} { r } { \mathbb { E } \left( \zeta \psi ( Y , y ) | X = x \right) = } \end{array}$ $m _ { \psi } ( x , y ) \mathbb { E } ( \zeta \mid X = x )$ . Thus we have

$$
m _ { \psi } ( x , y ) = { \frac { \operatorname { \mathbb { E } } ( \zeta \psi ( Y , y ) \mid X = x ) } { \operatorname { \mathbb { E } } ( \zeta \mid X = x ) } } .
$$

Given a random sample $( X _ { t } , Y _ { t } , \zeta _ { t } ) _ { 0 \leq t \leq T }$ one can therefore define a kernel-type estimator of $m _ { \psi } ( x , y )$ , say $\widehat { m } _ { \psi , T } ( x , y )$ , adapted to the MAR response framework. Note that if there are missing observations in the response variable, a simple way to estimate $m _ { \psi } ( x , y )$ is to consider a kernel smoothing-type estimator which only considers observed data, in other words, those for which $\zeta _ { t } = 1$ . Therefore, one gets

$$
\widehat { m } _ { \psi , T } ( \boldsymbol { x } , \boldsymbol { y } ) : = \left\{ \begin{array} { l l } { \displaystyle \frac { \int _ { 0 } ^ { T } \zeta _ { t } \psi \left( Y _ { t } , \boldsymbol { y } \right) \Delta _ { t } ( \boldsymbol { x } ) \mathrm { d } t } { \int _ { 0 } ^ { T } \zeta _ { t } \Delta _ { t } ( \boldsymbol { x } ) \mathrm { d } t } , } & { \mathrm { i f ~ } \displaystyle \int _ { 0 } ^ { T } \zeta _ { t } \Delta _ { t } ( \boldsymbol { x } ) \mathrm { d } t \neq 0 } \\ { \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } \zeta _ { t } \psi \left( Y _ { t } , \boldsymbol { y } \right) \mathrm { d } t , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

where $\begin{array} { r } { \Delta _ { t } ( x ) = K ( \frac { D _ { t } } { h _ { T } } ) , K ( \cdot ) } \end{array}$ is a kernel density function, $h _ { T }$ is the smoothing parameter tending to zero as $T$ goes to infinity.

Remark 2.1: When the sample has missing observations in the response variable, two strategies can be followed to estimate $m _ { \psi } ( x , y )$ . The first one, $\widehat { m } _ { \psi , T } ( x , y )$ given in Equation (3), called simplified estimator which only uses complete observations. The second approach consists in using the simplified estimator $\widehat { m } _ { \psi , T } ( x , y )$ to impute the missing values of the response variable $Y _ { t }$ according to the following expression: $\widetilde { \psi } ( Y _ { t } , y ) : =$ $\zeta _ { t } \psi ( Y _ { t } , y ) + ( 1 - \zeta _ { t } ) \widehat { m } _ { \psi , T } ( X _ { t } , y )$ , (see, e.g. Chu and Cheng 2003 or González-Manteiga and Pérez-González 2004). Consequently, an estimator, say $\widetilde { m } _ { \psi , T } ( x , y )$ , based on imputed data may be defined as follows:

$$
\widetilde { m } _ { \psi , T } ( \boldsymbol { x } , \gamma ) = \frac { \int _ { 0 } ^ { T } \widetilde { \psi } ( Y _ { t } , \gamma ) \Delta _ { t } ( \boldsymbol { x } ) \mathrm { d } t } { \int _ { 0 } ^ { T } \Delta _ { t } ( \boldsymbol { x } ) \mathrm { d } t } .
$$

From now on, we set $\begin{array} { r } { Z _ { 1 } ( x ) : = \int _ { 0 } ^ { \delta } \Delta _ { t } ( x ) } \end{array}$ dt and define the conditional bias as

$$
B _ { T } ( x , y ) : = \frac { \overline { { { m } } } _ { \psi , T , 2 } ( x , y ) } { \overline { { { m } } } _ { \psi , T , 1 } ( x ) } - m _ { \psi } ( x , y ) : = C _ { T } ( x , y ) - m _ { \psi } ( x , y ) ,
$$

where $\overline { { { m } } } _ { \psi , T , 1 } ( x ) : = \overline { { { m } } } _ { \psi , T , 1 } ( x , 1 )$ and for $i = 1 , 2$ ,

$$
\overline { { m } } _ { \psi , T , i } ( x , y ^ { i - 1 } ) : = \frac { 1 } { n \mathbb { E } ( Z _ { 1 } ( x ) ) } \int _ { 0 } ^ { T } \mathbb { E } \left\{ \zeta _ { t } ( \psi ( Y _ { t } , y ) ) ^ { i - 1 } \Delta _ { t } ( x ) \vert \mathcal { F } _ { t - \delta } \right\} \mathrm { d } t .
$$

Before introducing the assumptions under which we establish our asymptotic results, we add the following notations. Let $o _ { a . s . } ( u )$ denote a real random function $\ell$ such that $\ell ( u ) / u$ converges to zero almost surely (a.s.) as $u \to 0$ and denote $\mathcal { O } _ { a . s . } ( u )$ a real random function $\ell$ such that $\ell ( u ) / u$ is almost surely bounded.

(A1) (Assumptions on the kernel function). Let $K$ be a nonnegative bounded kernel of class ${ \mathcal { C } } ^ { 1 }$ over its support [0, 1] such that $K ( 1 ) > 0$ . The derivative $K ^ { \prime }$ exists on $[ 0 , 1 )$ and satisfies the condition $K ^ { \prime } ( \nu ) < 0$ for all $\nu \in [ 0 , 1 )$ and $\begin{array} { r } { | \int _ { 0 } ^ { 1 } ( K ^ { j } ) ^ { \prime } ( \nu ) \mathrm { d } \nu | < \infty } \end{array}$ for $j = 1 , 2$ .   
(A2) (Assumptions related to the continuous-time functional ergodic processes) Let $\scriptstyle { a _ { 0 } }$ be a nonnegative real number and $x \in { \mathcal { E } }$ . Suppose, for any $0 \leq s < t \leq$ $T$ such that $t - s \leq \alpha _ { 0 }$ , there exists a nonnegative continuous random function $f _ { t , s } ( x ) : = f _ { X _ { t } , s } ( x )$ a.s. bounded by a deterministic function $b _ { s , \alpha _ { 0 } } ( x ) ^ { 1 }$ .

Moreover, let $g _ { t , s , x } ( \cdot )$ be a random function defined on $\mathbb { R } , f ( x )$ is a deterministic nonnegative bounded function and $\phi ( \cdot )$ a nonnegative real function tending to zero (as its argument tends to 0), and assume that:

(i) $F _ { x } ( u ) : = \mathbb { P } ( d ( x , X _ { t } ) \leq u ) = \phi ( u ) f ( x ) + o ( \phi ( u ) )$ as $u \to 0$ .   
(ii) For any $\cdot 0 \le s \le t , F _ { x } ^ { \mathcal { F } _ { s } } ( u ) : = \mathbb { P } ^ { \mathcal { F } _ { s } } ( d ( x , X _ { t } ) \le u ) = \mathbb { P } ( d ( x , X _ { t } ) \le u \mid \mathcal { F } _ { s } ) = \phi ( u )$ $f _ { t , s } ( x ) + g _ { t , s , x } ( u )$ with $g _ { t , s , x } ( u ) = o _ { a . s . } ( \phi ( u ) )$ as $u \to 0$ , $g _ { t , s , x } ( u ) / \phi ( u )$ a.s. bounded and $\begin{array} { r } { T ^ { - 1 } \int _ { 0 } ^ { T } g _ { t , t - \delta , x } ( u ) \mathrm { d } t = o _ { a . s . } ( \phi ( u ) ) } \end{array}$ as $T \to \infty$ and $u \to 0$ .   
(iii) For any $\begin{array} { r } { x \in \mathcal { E } { : } \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } f _ { t , t - \delta } ( x ) \mathrm { d } t = f ( x ) } \end{array}$ , a.s.   
(iv) There exists a nondecreasing bounded function $\tau _ { 0 } : [ 0 , 1 ] \to [ 0 , 1 ]$ such that, uniformly in $u \in [ 0 , 1 ]$ ,

$$
\frac { \phi ( h u ) } { \phi ( h ) } = \tau _ { 0 } ( u ) + o ( 1 ) \quad \mathrm { a s \ } h \downarrow 0 \quad \mathrm { a n d } \quad \int _ { 0 } ^ { 1 } ( K ( \nu ) ) ^ { \prime } \tau _ { 0 } ( \nu ) \mathrm { d } \nu < \infty .
$$

(v) $\begin{array} { r } { T ^ { - 1 } \int _ { 0 } ^ { T } b _ { t , \alpha _ { 0 } } ( x ) \mathrm { d } t  D _ { \alpha _ { 0 } } ( x ) < \infty } \end{array}$ as $T \to \infty$ (A3) (Local smoothness and continuity conditions) Suppose for any $( y , t ) \in S \times [ 0 , T ]$ and $r > 0$ such that $t \leq r \leq t + \delta$ :

(i) ${ \mathbb E } ( \psi ( Y _ { r } , y ) | { \mathcal S } _ { t , \delta } ) = { \mathbb E } ( \psi ( Y _ { r } , y ) | X _ { r } ) = m _ { \psi } ( X _ { r } , y )$ a.s.   
$\mathrm { ( i ^ { \prime } ) }$ $\begin{array} { r } { \mathbb { E } ( \zeta _ { r } \psi ( Y _ { r } , \ y ) | S _ { t , \delta } ) = \mathbb { E } ( \zeta _ { r } \psi ( Y _ { r } , \ y ) | X _ { r } ) } \end{array}$ a.s. and for $\kappa \geq 2$ , $\mathbb { E } ( \zeta _ { r } | \psi ( Y _ { r } , \boldsymbol { y } ) | ^ { \kappa } | S _ { t , \delta } ) = \mathbb { E } ( \zeta _ { r } | \psi ( Y _ { r } , \boldsymbol { y } ) | ^ { \kappa } | X _ { r } )$ a.s.   
(ii) $\exists \beta > 0$ and a constant $c > 0$ such that, for any $( x ^ { \prime } , x ^ { \prime \prime } ) \in \mathcal { E } ^ { 2 }$ , $| m _ { \psi } ( x ^ { \prime } , y ) -$ $m _ { \psi } ( x ^ { \prime \prime } , y ) | \leq c d ^ { \beta } ( x ^ { \prime } , x ^ { \prime \prime } )$ .   
(iii) For any $\kappa _ { 1 } \geq 2$ , $\mathbb { E } ( | \psi ( Y _ { r } , \boldsymbol { y } ) | ^ { \kappa _ { 1 } } | S _ { t , \delta } ) = \mathbb { E } ( | \psi ( Y _ { r } , \boldsymbol { y } ) | ^ { \kappa _ { 1 } } | X _ { r } )$ a.s. The functions $W _ { \kappa _ { 1 } } ( x , y ) : = \mathbb { E } ( \vert \psi ( Y , y ) \vert ^ { \kappa _ { 1 } } \vert X = x )$ and $\overline { { W } } _ { \kappa _ { 1 } } ( x , y ) : = \mathbb { E } ( | \psi ( Y , y )$ $- m _ { \psi } ( x , y ) | ^ { \kappa _ { 1 } } | X = x )$ are continuous in the neighbourhood of $x$ and $\operatorname { s u p } _ { x \in { \mathcal { C } } , y \in S }$ $| W _ { \kappa _ { 1 } } ( x , y ) | < \infty$ a.s.   
(iv) $\mathbb { E } ( \zeta _ { r } | S _ { t , \delta } ) = \mathbb { E } ( \zeta _ { r } | X _ { r } ) = p ( X _ { r } )$ a.s. and for any $x \in { \mathcal { E } }$ , $\begin{array} { r } { \operatorname* { s u p } _ { \{ z : d ( x , z ) \leq u \} } | p ( z ) - p ( x ) | = o ( 1 ) } \end{array}$ a.s. as $u \to 0$ .

For $j = 1 , 2$ , define the following moments, which are independent of $x \in { \mathcal { E } }$

$$
M _ { j } = K ^ { ( j ) } ( 1 ) - \int _ { 0 } ^ { 1 } \left( K ^ { j } \right) ^ { \prime } ( u ) \tau _ { 0 } ( u ) \mathrm { d } u ,
$$

where $K ^ { ( j ) } ( \cdot )$ denotes the jth derivative of the kernel $K ( \cdot )$ and $( K ^ { j } ) ^ { \prime }$ the first derivative of $K$ raised to the power $j$ .

# 2.1. Comments on the assumptions

Condition (A1) is related to the choice of the kernel $K ,$ which is very usual in nonparametric functional estimation. Note that Parzen symmetric kernel is not adequate in this context since the random process $D _ { t } = d ( x , X _ { t } )$ is positive, therefore we consider $K$ with support [0, 1]. This is a natural generalisation of the assumption usually made on the kernel in the multivariate case where $K$ is supposed to be a spherically symmetric density function. The assumptions $K ( 1 ) > 0$ and $K ^ { \prime } < 0$ guarantee that $M _ { 1 } > 0$ for all limit functions $\tau _ { 0 }$ . In the case of non-smooth processes, $\tau _ { 0 }$ may be equal to the Dirac $\delta$ -function at 1, the condition

$K ( 1 ) > 0$ is needed to define the moments $M _ { j }$ which are, in this case, determined by the value $K ( 1 )$ .

Conditions (A2)(i) –(ii) reflect the ergodicity property assumed on the continuous-time functional process. It plays an important role in studying the asymptotic properties of the estimator. The functions $f _ { t , s }$ and $f$ play the same role as the conditional and unconditional densities in finite dimensional case, whereas $\phi ( u )$ characterises the impact of the radius $u$ on the small ball probability as $u$ goes to 0. Several examples to satisfy these conditions are given in Laïb and Louani (2010) for discrete-time functional data process. Some other examples satisfying this condition are also given in Didi and Louani (2014) when observations $( X _ { t } , Y _ { t } ) _ { 0 \leq t \leq T }$ are sampled from an ergodic continuous-time process taking values in $\mathbb { R } ^ { d } \times \mathbb { R }$ space.

Condition (A2)-(iii) involves the ergodic nature of the process where the random function $f _ { t , t - \delta }$ belongs to the space of continuous functions. Note that approximating the integral $\textstyle \int _ { 0 } ^ { T } f _ { t , t - \delta } ( x ) \mathrm { d } t$ by its Riemann’s sum: $\begin{array} { r } { T ^ { - 1 } \int _ { 0 } ^ { T } f _ { t , t - \delta } ( x ) \mathrm { d } t \simeq n ^ { - 1 } \sum _ { j = 1 } ^ { n } f _ { j \delta , ( j - 1 ) \delta } ( x ) } \end{array}$ allows to easily prove that the sequence $( f _ { j \delta , ( j - 1 ) \delta } ( x ) ) _ { j \geq 1 }$ is stationary and ergodic (see Didi and Louani 2014). (A2)-(iv) is a usual condition when dealing with functional data, whereas (A2)-(v) is a consequence of ergodic assumption.

(A.3)(ii) is a Hölder-type assumption that requires a certain smoothness of the regression operator $m _ { \psi } ( \cdot , y )$ . Such assumption is commonly used in nonparametric estimation. (A.3)(iii) is a smoothness condition on the $\kappa$ th centred conditional moments of $\psi ( Y , y )$ . (A3)(iv) assumes the continuity of the conditional probability of observing a missing response. Finally, note that the moments $( M _ { j } ) _ { j = 1 , 2 }$ are linked to the small probability function through $\tau _ { 0 }$ . One can refer to Ferraty et al. (2007) for a discussion on the choice of $\tau _ { 0 }$ , the Kernel $K$ and the positively of $( M _ { j } ) _ { j = 1 , 2 }$ .

Discussion on the assumptions $( A 3 ) ( i ) – ( i ^ { \prime } )$ . These hypotheses are Markov-type conditions that characterise the conditional moments of $\psi ( Y , y )$ . They are satisfied for a general class of processes including the $\alpha$ -mixing and non $\alpha$ -mixing as well as long memory and the Bernoulli shift processes. As pointed in Doukhan and Louhichi (1999), the main attraction of Bernoulli shift processes is that they provide examples of processes that are weakly dependent, but not mixing. According to the discussion made in the introduction we consider below some examples in both context (continuous and discretised processes) where the predictor $X$ is a stationary and ergodic Markovian process that might be $\alpha$ -mixing or not and satisfies the conditions (A.3)(i) $- ( i ^ { \prime } )$ .

First of all let us recall the following definitions.

Definition 2.1 (see Doukhan and Louhichi (1999)): Let $( \epsilon _ { i } ) _ { i \in \mathbb { Z } }$ be a sequence of independent real-valued r.v.s and $F$ be a measurable function defined on $\mathbb { R } ^ { \mathbb { Z } }$ . A Bernoulli shift is a sequence $( U _ { i } ) _ { i \in \mathbb { Z } }$ defined by $U _ { i } = F ( \epsilon _ { i - j } , j \in \mathbb { Z } )$ .

Definition 2.2 (see Doukhan $( { \bf 2 0 1 8 } , { \bf p . 6 0 } ) _ { . } ^ { \prime }$ ): A centred second-order stationary process $\left( X _ { n } \right)$ is called long-range dependent (LRD) if $\textstyle \sum _ { k = 0 } ^ { \infty } r _ { k } ^ { 2 } < \infty$ and $\begin{array} { r } { \sum _ { k = 0 } ^ { \infty } \left| r _ { k } \right| = \infty } \end{array}$ , where $r _ { k } = \mathrm { C o v } ( X _ { k } , X _ { 0 } )$ .

Definition 2.3: A process $( B _ { t } ^ { H } ) _ { t \in \mathbb { R } }$ is called a fractional Brownian motion $\left( \mathrm { f B m } \right)$ with Hurst exponent $H \in ( 0 , 1 ]$ ; if it is almost surely continuous, centred Gaussian process with

covariance

$$
\Gamma _ { H } ( s , t ) = \operatorname { C o v } \left( B _ { t } ^ { H } , B _ { s } ^ { H } \right) = \frac { 1 } { 2 } \left( | t | ^ { 2 H } + | s | ^ { 2 H } ( | t - s | ^ { 2 H } \right) , \quad \forall s , t \in \mathbb { R } .
$$

Definition 2.4 (see Lemma 4.2 in Maslowski and Pospíšil (2008)): A strictly stationary centred Gaussian process $( Y _ { t } ) _ { t \geq 0 }$ is ergodic if $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } R ( t ) : = \operatorname* { l i m } _ { t \to \infty } \mathbb { E } ( Y ( 0 ) Y ( t ) ) = 0 } \end{array}$ .

Example 2.5 (Continuous-time long memory processes): Let $\lambda , \sigma > 0$ and consider the Langevin equation with fBM noise $B _ { t } ^ { H }$ and initial condition $Z _ { 0 }$ :

$$
Z _ { t } = Z _ { 0 } - \lambda \int _ { 0 } ^ { t } Z _ { s } \mathrm { d } s + \sigma B _ { t } ^ { H } , \quad t \geq 0 .
$$

Then, for each $H \in ( 0 , 1 )$ , the following Gaussian stationary Markovian fractional Ornstein–Uhlenbeck process $( X _ { t } ^ { H } ) _ { t \geq 0 }$ defined as

$$
X _ { t } ^ { H } : = \sigma \int _ { - \infty } ^ { t } e ^ { - \lambda ( t - u ) } \mathrm { d } B _ { u } ^ { H } , \quad t \ge 0 ,
$$

is the unique (a.s.) solution of (7) with initial condition $Z _ { 0 } = X _ { 0 } ^ { H }$ (for more details, see Section 2, p. 5, in Cheridito et al. 2003).

Note that for $H \in ( \frac { 1 } { 2 } , 1 )$ , the auto-covariance function of $( X _ { t } ^ { H } ) _ { t \geq 0 }$ is similar to that of the increments of $( B _ { t } ^ { H } ) _ { t \geq 0 } ^ { - }$ . Therefore, $X _ { t } ^ { H }$ is ergodic (by Definition 2.3) and exhibits longrange dependence as detailed in Theorem 2.3 and the discussion in the end of page 8 in Cheridito et al. (2003).

Now, to check the condition (A.3)(i), consider the model:

$\begin{array} { r } { \psi ( Y _ { t } , y ) = m _ { \psi } ( X _ { t } ^ { H } , y ) + \epsilon _ { t } , } \end{array}$ where $\epsilon _ { t }$ ’s are centred, i.i.d. and independent of $X _ { t } ^ { H }$ .

Let $S _ { t , \delta }$ be the $\sigma$ -field generated by $\sigma ( ( X _ { s } ^ { H } , \epsilon _ { s } )$ , $( X _ { r } ^ { H } ) : 0 \leq s < t$ , $t \leq r \leq t + \delta )$ . It follows that, for any $r \geq t , \mathbb { E } [ \psi ( Y _ { r } , y ) \vert \mathcal { S } _ { t , \delta } ] = \mathbb { E } [ m _ { \psi } ( X _ { r } ^ { H } , y ) + \epsilon _ { r } \vert \mathcal { S } _ { t , \delta } ]$ .

Since $( X _ { r } ^ { H } , \epsilon _ { r } )$ are Markovian then $\mathbb { E } [ \psi ( Y _ { r } , y ) | S _ { t , \delta } ] = m ( X _ { r } ^ { H } )$ almost surely. Thus condition (A.3)(i) is satisfied.

Example 2.6 (Discrete-time processes): As discussed above, in real life we do not observe the process continuously at any time $t \in [ 0 , T ]$ . We rather observe a discretised version of it based on some sampling scheme. The following examples are used to show that Assumption (A3)(i) is also satisfied for discretised processes as well.

(i) Long-memory discrete-time processes. Let $( \epsilon _ { t } ) _ { t \in \mathbb { Z } }$ be a white noise process with variance $\sigma ^ { 2 }$ , and let $I$ and $B$ be the identity operator and the backshift operator, respectively. Giraitis and Leipus (1995) have proved (see Theorem 1 p. 55) that the $k$ -factor Gegenbauer process

$$
\prod _ { i \leq i \leq k } ( I - 2 \nu _ { i } B + B ^ { 2 } ) ^ { d _ { i } } X _ { t } = \epsilon _ { t } ,
$$

where $0 < d _ { i } < 1 / 2$ if $| \nu _ { i } | < 1$ or $0 < d _ { i } < 1 / 4$ if $| \nu _ { i } | = 1$ , for $i = 1 , \ldots , k ,$ is long memory, stationary, causal and invertible and has the moving average representation. That is $X _ { t } =$ $\textstyle \sum _ { j \geq 0 } \psi _ { j } ( d , \nu ) \epsilon _ { t - j }$ with $\textstyle \sum _ { j = 0 } ^ { \infty } \psi _ { j } ^ { 2 } ( d , \nu ) < \infty$ .

On the other hand, Guégan and Ladoucette (2001) have shown that, if $( \epsilon _ { t } ) _ { t \in \mathbb { Z } }$ is a Gaussian process, then the above process is not strong mixing whereas the moving average representation of $\left( X _ { t } \right)$ confirms that it is a stationary Gaussian and ergodic process.

(ii) The stationary solution of the linear Markov AR(1) process: $\begin{array} { r } { X _ { i } = \frac { 1 } { 2 } X _ { i - 1 } + \epsilon _ { i } , } \end{array}$ where $( \epsilon _ { i } )$ are independent symmetric Bernoulli random variables taking values $- 1$ and 1, is not $\alpha$ -mixing (see Andrews 1984). However, $( X _ { i } )$ is a Markovian stationary and ergodic process. (iii) Let $\left( u _ { i } \right)$ be an i.i.d. sequence uniformly distributed on $\{ 1 , . . . , 9 \}$ , and set $X _ { t } : =$ $\scriptstyle \sum _ { i = 0 } ^ { \infty } 1 0 ^ { - i - 1 } u _ { t - i } ,$ where the sequence $u _ { t } , u _ { t - 1 } , \ldots ,$ represents the decimals of $X _ { t }$ . The process $X = \left( X _ { t } \right)$ is stationary and admits the following AR(1) representation: $\begin{array} { r } { X _ { t } = \frac { 1 } { 1 0 } X _ { t - 1 } + } \end{array}$ $\begin{array} { r } { \frac { 1 } { 1 0 } u _ { t } = \frac { 1 } { 1 0 } X _ { t - 1 } + \frac { 1 } { 2 } + \epsilon _ { t } } \end{array}$ where $\begin{array} { r } { \epsilon _ { t } = \frac { 1 } { 1 0 } u _ { t } - \frac { 1 } { 2 } } \end{array}$ is a strong white noise. This process is not $\alpha$ -mixing (see Francq and Zakoïan 2010, Example A.3, p. 349), but it is ergodic.

To check the hypothesis (A3)(i) for Examples (i), (ii) and (iii), consider the regression model $\psi ( Y _ { i } , y ) = m _ { \psi } ( X _ { i } , y ) + \sigma _ { \psi } ( X _ { i } ) \eta _ { i } ,$ where $( \eta _ { i } )$ is a white noise process independent of $( X _ { i } )$ and define the $\sigma$ -field: $\mathcal { G } _ { i } = \sigma ( ( X _ { 1 } , \eta _ { 1 } , \zeta _ { 1 } ) , \dots , ( X _ { i } , \eta _ { i } , \zeta _ { i } ) , X _ { i + 1 } )$ . It is then easy to see that condition (A3)(i) is fulfilled. The discrete-time processes in examples (i)–(iii) are still valid for the regression model developed in Section 3.5 under the context of sampling schemes.

# 3. Main results

In this section, we investigate several asymptotic properties of the continuous-time generalised regression estimator. Some particular cases, related to specific choices of the function $\psi ( \cdot , y )$ , including the conditional cumulative distribution function and the conditional quantiles will also be discussed.

# 3.1. Almost sure consistency rates

# 3.1.1. Pointwise consistency

The following theorem establishes an almost sure pointwise consistency rate of $\widehat { m } _ { \psi , T } ( x , y )$

Theorem 3.1 (Pointwise consistency): Assume that (A1)–(A3) hold true and the following conditions are satisfied:

$$
\operatorname * { l i m } _ { T  \infty } T \phi ( h _ { T } ) = \infty \quad a n d \quad \operatorname * { l i m } _ { T  \infty } \frac { \log T } { T \phi ( h _ { T } ) } = 0 .
$$

Then, for $T$ sufficiently large, we have

$$
\widehat { m } _ { \psi , T } ( x , y ) - m _ { \psi } ( x , y ) = \mathcal { O } _ { a . s . } ( h _ { T } ^ { \beta } ) + \mathcal { O } _ { a . s . } \left( \sqrt { \frac { \log T } { T \phi ( h _ { T } ) } } \right) .
$$

The proof of Theorem 3.1 is detailed in the supplementary material in Chaouch and Laïb (2023).

Remark 3.1: Theorem 3.1 generalises Theorem 1 of Laïb and Louani (2011) established in the context of discrete-time stationary and ergodic processes, and Theorem 3.4 of Ferraty et al. (2005) stated under a strong mixing assumption with completely observed response where the support of y is reduced to one point. Moreover, the function $\phi ( h _ { T } )$ can decrease to zero at an exponential rate, whenever $h _ { T }$ goes to zero, therefore $h _ { T }$ should be chosen to decrease to zero at a logarithmic rate.

# 3.1.2. Uniform consistency

To establish the uniform consistency with rate of the regression operator, we need some additional definitions and assumptions that allow to express the uniform convergence rate as a function of the entropy number. Let $\mathcal { C }$ and S be compact sets in $\mathcal { E }$ and $\mathbb { R } ,$ respectively. Consider, for any $\epsilon > 0$ , the $\epsilon$ -covering number of the compact set $\mathcal { C }$ , say $N _ { \epsilon } : = \mathcal { N } ( \epsilon , \mathcal { C } , d )$ , defined by

$$
\begin{array} { r l } & { N _ { \epsilon } : = \operatorname* { m i n } \left\{ n : \mathrm { t h e r e ~ e x i s t } c _ { 1 } , . . . , c _ { n } \in \mathcal { C } \mathrm { ~ s u ~ } \right. } \\ & { ~ \left. 1 \leq i \leq n \mathrm { ~ f o r ~ w h i c h ~ } d ( x , c _ { i } ) < \epsilon \right\} . } \end{array}
$$

The number $N _ { \epsilon }$ measures how full is the class $\mathcal { C }$ . The finite set of points $c _ { 1 } , c _ { 2 } , \ldots , c _ { N _ { \epsilon } }$ is called an $\epsilon$ -net of $\mathcal { C }$ if $\mathcal { C } \subset \cup _ { k = 1 } ^ { N _ { \epsilon } } B ( c _ { k } , \epsilon )$ , where $B ( c _ { k } , \epsilon )$ is the ball, centred at $c _ { k }$ and of radius $\epsilon$ , with respect to the topology induced by the semi-metric $d ( \cdot , \cdot )$ . The quantity $\varphi _ { \mathcal { C } } ( \epsilon ) = \log ( N _ { \epsilon } )$ is called the Kolmogorov’s $\epsilon$ -entropy of the set $\mathcal { C }$ that may be seen as a tool to measure the complexity of the subset $\mathcal { C }$ , in the sense that high entropy means that a large amount of information is needed to describe an element of $\mathcal { C }$ with an accuracy $\epsilon$ . Several examples of $\varphi _ { \mathcal { C } } ( \epsilon )$ covering special cases of functional processes are given in Ferraty et al. (2010) and Laïb and Louani (2011).

(U0) Assume that (A2) holds uniformly in the following sense:

(i) (A2)(i) and (A2)(ii) hold true with the remaining term $o ( \phi ( u ) )$ is uniform in $x$ (ii) For any $x \in { \mathcal { C } }$ , $\begin{array} { r } { \operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { x \in \mathcal { C } } | \frac { 1 } { T } \int _ { 0 } ^ { T } f _ { t , t - \delta } ( x ) \mathop { } \dot { \mathrm { d } t } - f ( x ) | = 0 } \end{array}$ a.s. (iii) $\begin{array} { r } { T ^ { - 1 } \int _ { 0 } ^ { T } b _ { t , \alpha _ { 0 } } ( x ) \mathrm { d } t  D _ { \alpha _ { 0 } } ( x ) } \end{array}$ as $T \to \infty$ with $0 < \mathsf { s u p } _ { x \in \mathcal { C } } D _ { \alpha _ { 0 } } ( x ) < \infty$ . (iv)) $b _ { 0 } < \operatorname* { i n f } _ { x \in { \mathcal { C } } } f ( x ) \leq \operatorname* { s u p } _ { x \in { \mathcal { C } } } f ( x ) < \infty$ for some nonnegative real number $b _ { 0 }$ . (v) $\operatorname* { i n f } _ { x \in { \mathcal { C } } } p ( x ) > b _ { 1 }$ for some nonnegative real number $b _ { 1 }$ .

(U1) The kernel function $K$ satisfies the following conditions:

(i) K is a Hölder function of order 1 with a constant $a _ { K }$ .   
(ii) There exist two constants $a _ { 2 }$ and $_ { a _ { 3 } }$ such that $0 < a _ { 2 } \le K ( x ) \le a _ { 3 } < \infty$ for any $x \in { \mathcal { C } }$ .

(U2) For $1 \leq \ell \leq 2$ , the sequence of random variables $( \psi ^ { \ell } ( Y _ { t } , y ) ) _ { t }$ is ergodic and $\mathbb { E } ( | \psi ^ { \ell } ( Y _ { 0 } , y ) | ) < \infty$ .

(U3) There exist $c _ { \psi } > 0$ and nonnegative real number γ such that for any $y \in S$

$$
\operatorname* { s u p } _ { y ^ { \prime } \in [ y - u , y + u ] \cap S } | \psi ( Y _ { t } , y ) - \psi ( Y _ { t } , y ^ { \prime } ) | \leq c _ { \psi } u ^ { \gamma } .
$$

(U4) Let $T _ { n }$ be the integer part of $T$ and suppose for $T$ large enough

$$
\frac { ( \log T ) ^ { 2 } } { T \phi ( h _ { T } ) } < \varphi _ { C } ( \epsilon _ { n } ) < \frac { T \phi ( h _ { T } ) } { \log T } \quad \mathrm { w i t h } \ \epsilon _ { n } = \frac { \log T _ { n } } { T _ { n } } .
$$

Conditions in (U0) are standard in this context to get uniform consistency rate. Condition (U1) is usually used when we deal with nonparametric estimation for functional data, (U2) requires the existence of the moments up to order 2 of $\psi ( Y , y )$ . (U3) is a regularity condition upon the function $\psi ( \cdot , y )$ which is necessary to obtain the uniform consistency result over the compact S. (U4) allows to cover the subset $\mathcal { C }$ with a finite number of balls and to express the convergence rate in terms of the Kolmogorov’s entropy of this subset. Similar condition has been used in Ferraty et al. (2010), where the authors have pointed out that, for a radius not too large, one requires the quantity $\varphi _ { C } ( \log T _ { n } / T _ { n } )$ to be not too small and not too large. This condition seems to satisfy this exigence, since it implies that $\frac { \varphi _ { C } ( \log T _ { n } / T _ { n } ) } { T \phi ( h _ { T } ) }$ goes to 0 for sufficiently large $T$ . Examples given in Ferraty et al. (2010) and Laïb and Louani (2011) satisfy (U4).

Theorem 3.2 states uniform consistency rate of the Kernel regression estimator. It generalises Theorem 2 in Ferraty et al. (2010) in the i.i.d. case and that in Laïb and Louani (2011) established in the context of discrete-time stationary and ergodic processes with completely observed response.

Theorem 3.2 (Uniform consistency): Assume (A1), (U0)–(U4), (A3) hold true. Moreover, suppose conditions in (8) are satisfied and

$$
\sum _ { n \geq 1 } n ^ { \gamma } \exp \left\{ ( 1 - \eta ) \varphi _ { \mathcal { C } } \left( { \frac { \log n } { n } } \right) \right\} < \infty \quad { \mathrm { f o r s o m e ~ } } \eta > 0 { \mathrm { ~ w h e r e ~ } } \gamma { \mathrm { ~ i s a s i n ~ } } ( U 3 ) .
$$

Then we have

$$
\operatorname* { s u p } _ { \substack { \ell \in S } } | \widehat { m } _ { \psi , T } ( x , y ) - m _ { \psi } ( x , y ) | = \mathcal { O } _ { a . s . } ( h _ { T } ^ { \beta } ) + \mathcal { O } _ { a . s . } ( \sqrt { \frac { \varphi _ { \mathcal { C } } ( \epsilon _ { T } ) } { T \phi ( h _ { T } ) } } ) \quad \mathrm { a s } T  + \infty .
$$

The proof of Theorem 3.2 is detailed in the supplementary material in Chaouch and Laïb (2023).

# 3.2. Asymptotic conditional bias and risk evaluation

Before evaluating the conditional bias, let us introduce some additional notations. Consider for $i = 1 , 2$ , the following assumption:

(BC1) Recall that $D _ { t } ( x ) = d ( X _ { t } , x )$ and, for any $t \geq 0$ , denote

$$
\begin{array} { r l } & { \mathbb { E } \left[ m _ { \psi } ( X _ { t } , y ) - m _ { \psi } ( x , y ) | D _ { t } ( x ) , \mathcal { F } _ { t - \delta } \right] } \\ & { \quad = \mathbb { E } \left[ m _ { \psi } ( X _ { t } , y ) - m _ { \psi } ( x , y ) | D _ { t } ( x ) \right] = : \Psi _ { y } ( D _ { t } ( x ) ) . } \end{array}
$$

Assume that the function $\Psi _ { y }$ is differentiable at 0 and satisfies $\Psi _ { y } ( 0 ) = 0$ and $\Psi _ { y } ^ { \prime } ( 0 ) \neq 0$ for any $\boldsymbol { y } \in \mathbb { R }$ . This condition was introduced in Ferraty et al. (2007) and used by Laïb and Louani (2010) to evaluate the conditional bias. The introduction of $\Psi _ { y } ( \cdot )$ allows to make an integration with respect to the real random variable $D _ { t } ( x )$ rather than the couple of random variables $( D _ { t } ( x ) , X _ { t } )$ , where $X _ { t }$ being functional continuous random variable.

The following proposition gives asymptotic expression of the conditional bias term, which generalises Proposition 1 in Laïb and Louani (2010) for discrete-time estimator to our setting. Its proof is similar to the one in the discrete-time framework and therefore is omitted.

Proposition 3.3 (Conditional Bias): Under assumptions (A1)–(A3), (BC1) and conditions in (8), we have

$$
\Gamma ( x , y ) = \frac { h _ { T } \Psi _ { y } ^ { \prime } ( 0 ) } { M _ { 1 } } \left[ K ( 1 ) - \int _ { 0 } ^ { 1 } ( s K ( s ) ) ^ { \prime } \tau _ { 0 } ( s ) \mathrm { d } s + o _ { a , s } ( 1 ) \right] + \mathcal { O } _ { a , s } \left( h _ { T } \sqrt { \frac { \log T } { T \phi ( h _ { T } ) } } \right) .
$$

The next result gives an explicit expression of the asymptotic quadratic risk associated to the estimator $\widehat { m } _ { \psi , T } ( x , y )$ .

Theorem 3.4 (Quadratic risk): Suppose that Assumptions (A1)–(A3) hold true. Then, whenever $p ( x ) > 0$ and $f ( x ) > 0$ , we have, for a fixed $( x , y ) \in \mathcal { E } \times \mathbb { R } ,$ that

$$
\begin{array} { r l } & { M S E ( x , y ) : = \mathbb { E } \left[ \left( \widehat { m } _ { \psi , T } ( x , y ) - m _ { \psi } ( x , y ) \right) ^ { 2 } \right] } \\ & { = A _ { 1 } h _ { T } ^ { 2 } \left[ A _ { 1 } + \mathcal { O } \left( \sqrt { \frac { \log T } { T \phi ( h _ { T } ) } } \right) \right] + \frac { A _ { 2 } ( x , y ) } { T \phi ( h _ { T } ) } , } \end{array}
$$

where

$$
\begin{array} { r l } & { { \cal A } _ { 1 } = \displaystyle \frac { \Psi _ { y } ^ { \prime } ( 0 ) } { M _ { 1 } } \left[ K ( 1 ) - \int _ { 0 } ^ { 1 } ( s K ( s ) ) ^ { \prime } \tau _ { 0 } ( s ) \mathrm { d } s + o ( 1 ) \right] \quad \mathrm { a n d } } \\ & { { \cal A } _ { 2 } ( x , y ) = \displaystyle \frac { 4 \left( W _ { 2 } ( x , y ) + ( m _ { \psi } ( x , y ) ) ^ { 2 } \right) M _ { 2 } } { p ( x ) M _ { 1 } ^ { 2 } f ( x ) } . } \end{array}
$$

Remark 3.2: (i) Note that, for sufficiently large $T ,$ the expression of MSE becomes $A _ { 1 } ^ { 2 } h _ { T _ { + } } ^ { 2 } + \frac { A _ { 2 } ( x , y ) } { T \phi ( h _ { T } ) }$ . This result generalises the one in Chaouch and Laïb (2019) established in the framework of real-valued continuous-time processes. Note, however that, for finite-dimensional continuous-time processes with MAR response, the bias term obtained in Chaouch and Laïb (2019) is of order $h _ { T } ^ { 2 }$ which is smaller than $h _ { T }$ given in Proposition 3.3. The increase in the bias term is because of the infinite dimensional characteristic of the functional space.

(ii) The mean squared error can be used as a theoretical guidance to select the ‘optimal’ bandwidth by minimising the quantity A21h2T + A2(x,y)Tφ(hT) with respect to $h _ { T }$ . However, $A _ { 1 }$ and $A _ { 2 } ( x , y )$ depend on some unknown quantities which should be replaced by their empirical consistent estimators, namely $\Psi _ { y , T } ^ { \prime } ( 0 )$ , $( M _ { j , T } ) _ { j = 1 , 2 }$ , $\tau _ { 0 , T } , \ p _ { T } , W _ { 2 , T }$ , and $f _ { T }$ . Note that $\Psi _ { y } ^ { \prime } ( 0 )$ may be viewed as real regression function with response variable $m _ { \psi } ( X , y ) \stackrel { , } { - } m _ { \psi } ( x , y )$ and predictor $d ( X , x )$ . It may be then estimated by a kernel regression estimate $\Psi _ { \gamma , T } ^ { \prime } ( 0 )$ by replacing $m _ { \psi } ( x , y )$ by its estimator $\widehat { m } _ { \psi , T } ( X , y )$ .

# 3.3. Asymptotic normality

The following theorem establishes the asymptotic distribution of the estimator.

Theorem 3.5: Assume that conditions (A1)–(A3) are fulfilled. Suppose that, for $\beta$ as defined in (A3)(ii), the following conditions hold true:

$$
\operatorname * { l i m } _ { T \to \infty } T \phi ( h _ { T } ) = \infty , \quad h _ { T } ^ { \beta } \sqrt { T \phi ( h _ { T } ) } = o ( 1 ) \quad \mathrm { a n d } \quad h _ { T } ^ { \beta } \log T ^ { 1 / 2 } = o ( 1 ) \quad \mathrm { a s } T \to \infty
$$

Then, for any $( x , y ) \in \mathcal { E } \times S$ such that $f ( x ) > 0$ , we have

$$
\sqrt { T \phi ( h _ { T } ) } ( \widehat { m } _ { \psi , T } ( x , y ) - m _ { \psi } ( x , y ) ) \stackrel { d } { \to } \ N ( 0 , \sigma ^ { 2 } ( x , y ) ) ,
$$

where

$$
\sigma ^ { 2 } ( x , y ) \leq \frac { 1 } { f ( x ) } \frac { M _ { 2 } } { M _ { 1 } ^ { 2 } { \widehat { p } } ( x ) } \overline { { W } } _ { 2 } ( x , y ) = : \frac { 1 } { f ( x ) } \widetilde { V } ( x , y ) \quad \mathrm { a s } T \longrightarrow \infty
$$

and

$$
\overline { { W } } _ { 2 } ( x , y ) : = \mathbb { E } \left[ \left( \psi ( Y , y ) - m _ { \psi } ( x , y ) \right) ^ { 2 } | X = x \right] .
$$

Note that the statement (13) gives only an upper bound of the asymptotic variance $\sigma ^ { 2 } ( x , y )$ . The following proposition gives an estimate of $\widetilde { V } ( x , y )$ that will be needed to construct confidence intervals for the unknown operator $m _ { \psi } ( x , y )$ .

Proposition 3.6: Suppose conditions of Theorem 3.5 hold and $\sigma ^ { 2 } ( x , y ) > 0 .$ , then

$$
\widehat { V } _ { T } ( x , y ) : = \frac { \sqrt { M _ { T , 2 } } } { M _ { 1 , T } } \sqrt { \frac { \overline { { W } } _ { 2 , T } ( x , y ) } { T F _ { x , T } ( h _ { T } ) p _ { T } ( x ) } } ,
$$

is a consistent estimator for $\widetilde { V } ( x , y )$ . The quantities $M _ { 1 , T } , M _ { 2 , T } , \overline { { { W } } } _ { 2 , T } , p _ { T } ( x )$ and $F _ { x , T }$ are empirical versions of $M _ { 1 } , M _ { 2 } , \overline { { { W } } } _ { 2 } , p ( x )$ and $F _ { x }$ , respectively.

$M _ { 1 , T }$ and $M _ { 2 , T }$ are calculated by replacing $\tau _ { 0 }$ , given in (A2)(iv), by its empirical version

$$
\tau _ { 0 , T } ( u ) = \frac { F _ { x , T } ( u h _ { T } ) } { F _ { x , T } ( h _ { T } ) } \quad \mathrm { w h e r e ~ } F _ { x , T } ( u ) = \frac { 1 } { T } \int _ { 0 } ^ { T } \mathbb { 1 } _ { \{ d ( x , X _ { t } ) \leq u \} } \mathrm { d } t .
$$

On the other hand $\overline { { W } } _ { 2 , T } ( x , y )$ and $p _ { T } ( x )$ are given by

$$
\overline { { W } } _ { 2 , T } ( x , y ) = \frac { \int _ { 0 } ^ { T } \zeta _ { t } ( \psi ( Y _ { t } , y ) ) ^ { 2 } \Delta _ { t } ( x ) \mathrm { d } t } { \int _ { 0 } ^ { T } \zeta _ { t } \Delta _ { t } ( x ) \mathrm { d } t } - ( \widehat { m } _ { \psi , T } ( x , y ) ) ^ { 2 } \quad \mathrm { a n d } \quad \hat { p } _ { T } ( x ) = \frac { \int _ { 0 } ^ { T } \zeta _ { t } \Delta _ { t } ( x ) \mathrm { d } t } { \int _ { 0 } ^ { T } \Delta _ { t } ( x ) \mathrm { d } t }
$$

# 3.4. Continuous-time confidence intervals

Using the non-decreasing property of the cumulative standard Gaussian distribution function, the estimator $\tilde { \widehat { V } } _ { T } ( x , \bar { y } )$ , defined in (14), with the help of Proposition 3.6 and Theorem 3.5, the following corollary provides estimated confidence intervals for $m _ { \psi } ( x , y )$ at any $x$ fixed.

Corollary 3.7: Assume conditions of Theorem 3.5 are fulfilled and the conditions in (12) are replaced by

$$
\operatorname* { l i m } _ { T \to \infty } T F _ { x , T } ( h _ { T } ) = \infty \quad \mathrm { a n d } \quad \operatorname* { l i m } _ { T \to \infty } h _ { T } ^ { \beta } \sqrt { T F _ { x , T } ( h _ { T } ) } = 0 .
$$

Then, for any $0 < \alpha < 1 _ { : }$ , the $( 1 - \alpha )$ confidence intervals for $m _ { \psi } ( x , y )$ are

$$
I _ { 1 - \alpha } ( x , y ) = \left[ \widehat { m } _ { \psi , T } ( x , y ) - c _ { \alpha / 2 } \sqrt { \widehat { V } _ { T } ( x , y ) } ; \widehat { m } _ { \psi , T } ( x , y ) + c _ { 1 - \alpha / 2 } \sqrt { \widehat { V } _ { T } ( x , y ) } \right] ,
$$

where $c _ { \alpha }$ is the αth quantile of the standard normal distribution.

These intervals are similar to those given in Remark 2 in Laïb and Louani (2010) for discrete-time ergodic context with complete data, and those obtained in Ling et al. (2015) for discrete-time stationary ergodic data with missing at random response.

# 3.5. Sampling schemes and computation of the confidence intervals

In the previous section, the process was supposed to be observable over [0, T]. However, in practice the data are often collected according to a sampling scheme since it is difficult to observe a path continuously at any time $t$ over the interval [0, T]. Hereafter, we briefly discuss the effect of a sampling scheme on the construction of confidence intervals for the regression function $m _ { \psi } ( x , y )$ . Assume that the data are sampled, either regularly, irregularly or even randomly, from an underlying continuous-time process at instants $( t _ { k } ) _ { k = 1 , \dots , n }$ . For a sake of simplicity, we consider here the case where the instants $\left( t _ { k } \right)$ are irregularly spaced, that is $\begin{array} { r } { \operatorname* { i n f } _ { 1 \leq k \leq n } | t _ { k + 1 } - t _ { k } | = \delta > 0 . } \end{array}$ Now, for $k \in \{ 1 , \ldots n \}$ , we define the following increasing families of $\sigma$ -algebra:

$$
\mathcal { F } _ { k } : = \mathcal { F } _ { t _ { k } } = \sigma \left( ( X _ { t _ { 1 } } , Y _ { t _ { 1 } } ) , \ldots , ( X _ { t _ { k } } , Y _ { t _ { k } } ) \right)
$$

and

$$
\mathcal { G } _ { k } : = \mathcal { G } _ { t _ { k } } = \sigma \left( \left( X _ { t _ { 1 } } , Y _ { t _ { 1 } } \right) , \ldots , \left( X _ { t _ { k } } , Y _ { t _ { k } } \right) ; X _ { t _ { k + 1 } } \right) .
$$

The purpose then consists in estimating $m _ { \psi } ( x , y )$ given the discrete-time ergodic stationary process $( X _ { t _ { k } } , Y _ { t _ { k } } , \zeta _ { t _ { k } } ) _ { 1 \leq k \leq n }$ sampled from the underlying continuous-time process $\{ ( X _ { t } , Y _ { t } , \zeta _ { t } ) \} _ { 0 \leq t \leq T }$ . In case of a regular sampling scheme, that is $T = n \delta$ , the regression function is $\begin{array} { r } { m _ { \psi } ^ { \mathrm { ~ - ~ } } ( x , y ) : = \frac { \mathbb { E } ( \zeta _ { t _ { k } } \psi ( Y _ { t _ { k } } , y ) \bar { | } X _ { t _ { k } = x } ) } { \mathbb { E } ( \zeta | X _ { t _ { k } = x } ) } } \end{array}$ , $1 \leq k \leq n$ and its estimator $\widehat { m } _ { \psi , T } ( x , y )$ defined in (3) becomes

$$
\widehat { m } _ { \psi , n } ( x , y ) = \frac { \sum _ { k = 1 } ^ { n } \zeta _ { t _ { k } } \psi \left( Y _ { t _ { k } } , y \right) K \left( \frac { d ( x , X _ { t _ { k } } ) } { h _ { n } } \right) } { \sum _ { k = 1 } ^ { n } \zeta _ { t _ { k } } K \left( \frac { d ( x , X _ { t _ { k } } ) } { h _ { n } } \right) } , \quad t _ { k } = k \delta \left( 1 \leq k \leq n \right) .
$$

Note that Theorem 3.5 holds for the estimate $\widehat { m } _ { \psi , n } ( x , \gamma )$ when replacing $T$ by $n \delta$ . The limiting law is a Gaussian random variable with mean zero and variance function $\sigma ^ { 2 } ( x , y ) =$ $\frac { 1 } { f ( x ) } \frac { \ l } { M _ { 1 } ^ { 2 } p ( x ) } \overline { { { W } } } _ { 2 } ( x , y )$ . Making use of Corollary 3.7 and considering similar steps as in Laïb

and Louani (2010), it follows that, for any $0 < \alpha < 1$ , the $( 1 - \alpha )$ asymptotic confidence intervals of $m _ { \psi } ( x , y )$ are

$$
\widehat { m } _ { \psi , n } ( x , y ) \pm c _ { 1 - \alpha / 2 } \frac { \sqrt { M _ { n , 2 } } } { M _ { n , 1 } } \sqrt { \frac { \overline { { { W } } } _ { n , 2 } ( x , y ) } { n F _ { x , n } ( x ) p _ { n } ( x ) } } , \quad \mathrm { a s \ } n \to \infty ,
$$

where $c _ { 1 - \alpha / 2 }$ is the quantile of standard normal distribution.

# 4. Simulation study

This section aims to discuss numerically some aspects related to continuous-time processes that might affect the quality of estimation of the operator $m _ { \psi } ( x , y )$ . Here we consider $\psi ( Y _ { t } , y ) = Y _ { t }$ , therefore $m _ { \psi } ( x , y ) = m ( x ) = \mathbb { E } ( Y _ { t } | X _ { t } = x )$ , where $m ( x )$ is the conditional expectation of $Y _ { t }$ given $X _ { t } = x$ . The first simulation aims to compare the quality of estimation of $m ( x )$ based on the continuous-time and discrete-time processes. In the second simulation, we discuss the choice of the ‘optimal’ sampling mesh $\delta$ in the case of continuous-time processes and assess its sensitivity to the missing at random mechanism. Finally, the third simulation discusses the effect of the MAR rate on the coverage rate and length of the estimated confidence intervals.

# 4.1. Simulation 1: continuous-time versus discrete-time estimators

In this first simulation, we try to compare the estimation of the regression operator when discrete- and continuous-time processes are considered. We want to know whether considering a continuous-time processes may improve the quality of the predictions or not. We suppose that the functional space $\mathcal { E } = L ^ { 2 } ( [ - 1 , 1 ] )$ endowed with its natural norm. The generation of continuous-time processes $\{ X _ { t } ( s ) : s \in [ - 1 , 1 ] \} , Y _ { t } ) _ { t \in [ 0 , T ] }$ is obtained by considering the following steps:

(1) First, we simulate an Ornstein–Uhlenbeck (OU) process $( Z _ { t } ) _ { t \geq 0 }$ solution of the following stochastic differential equation:

$$
\mathrm { d } Z _ { t } = 2 ( 5 - Z _ { t } ) \mathrm { d } t + 7 \mathrm { d } \mathrm { W } _ { t } ,
$$

where $\mathbf { W } _ { t }$ denotes a Wiener process. Here, we take $\mathrm { d } t = 0 . 0 0 5$ .

(2) Let $\Gamma ( \cdot )$ be the operator mapping $\mathbb { R }$ into $L ^ { 2 } ( [ - 1 , 1 ] )$ defined, for any $z \in \mathbb { R }$ , as follows:

$$
\Gamma ( z ) : = ( 1 + \lfloor z \rfloor - z ) \mathrm { P } _ { \mathrm { n u m } ( \lfloor z \rfloor ) } + ( z - \lfloor z \rfloor ) \mathrm { P } _ { \mathrm { n u m } ( \lfloor z + 1 \rfloor ) } ,
$$

where $P _ { j }$ is the Legendre polynomials of degree $j$ and n $\operatorname { u m } ( z ) : = 1 + 2 z \operatorname { s i g n } ( z ) -$ $\mathrm { s i g n } ( z ) ( 1 + \mathrm { s i g n } ( z ) ) / 2$ and $\lfloor \cdot \rfloor$ denotes the floor function.

(3) We consider that curves are sampled at 400 equispaced values in $[ - 1 , 1 ]$ and defined, for any $t \in [ 0 , T ]$ , as

$$
X _ { t } ( s ) = \Gamma ( Z _ { t } ) ( s ) , \quad s \in [ - 1 , 1 ] .
$$

![](images/795698db5216aed4e185dd1d054a05236a89c3fa5001e8374f87b947e25c4aea.jpg)  
Left: A sample of 20 simulated curves $\{ X _ { t } ( s ) : s \in [ - 1 ,$ 1]}. Right: A realisation of the process $( Y _ { t } ) _ { t \in [ 0 , 2 0 0 ] }$ .

(4) To generate the real-valued process $( Y _ { t } ) _ { t \in [ 0 , T ] }$ , the following nonlinear functional regression model is considered:

$$
Y _ { t } = m ( X _ { t } ) + \varepsilon _ { t } ,
$$

where $\begin{array} { r } { m ( x ) : = \int _ { - 1 } ^ { 1 } x ^ { 2 } ( s ) \mathrm { d } } \end{array}$ s, $\varepsilon _ { t } = U _ { t } - U _ { t - 1 }$ where $U _ { t }$ is a Wiener process independent of $X _ { t }$ .

Observe that the OU process $\{ Z _ { t } : t \in [ 0 , T ] \}$ is a real-valued continuous-time process (since dt tends to zero). The operator $\Gamma ( \cdot )$ has a role to transform each observation in the process $Z _ { t }$ into a curve through the Legendre polynomials. In such way, the functional variable $X$ is generated continuously as is the process $\left( Z _ { t } \right)$ . Moreover, note that steps 1, 2 and 3 are devoted to simulate the continuous-time functional process $\{ X _ { t } ( s ) : s \in [ - 1 , 1 ] \} _ { t \in [ 0 , T ] }$ , whereas in step 4 the real-valued continuous-time process $( Y _ { t } ) _ { t \in [ 0 , T ] }$ is generated. A sample of 20 simulated curves is displayed in Figure 1(left) and an example of the real-valued process $( Y _ { t } )$ is given in Figure $1 ( \mathrm { r i g h t } )$ .

Now, our purpose is to compare, in terms of estimation accuracy, the continuous-time estimator with the discrete-time one for different values of $T = 5 0 , 2 0 0 , 1 0 0 0$ and several missing at random rates. It is worth noting that the continuous-time process $( X _ { t } , Y _ { t } )$ is observed at every instant $t = \delta , 2 \delta , \ldots , n \delta$ , where $\delta = 0 . 0 0 5$ and $n = T / \delta$ . However, the discrete-time process is observed only at the instants $t = 1 , 2 , \dots , n$ .

As in Ferraty et al. (2013) and Ling et al. (2015), we consider that the missing at random mechanism is led by the following probability distribution:

$$
p ( x ) = \mathbb { P } ( \zeta _ { t } = 1 | X _ { t } = x ) = \exp \mathrm { i t } \left( \int _ { - 1 } ^ { 1 } x ^ { 2 } ( s ) \mathrm { d } s \right) ,
$$

where ex $\mathrm { p i t } ( u ) = e ^ { u } / ( 1 + e ^ { u } ) .$ , for $u \in \mathbb { R }$ . Now, we specify the tuning parameters on which depend our estimator given in (3). We choose the quadratic kernel defined as $K ( u ) =$ $\textstyle { \frac { 3 } { 4 } } ( 1 - u ^ { 2 } ) \mathbb { 1 } _ { ( 0 , 1 ) } ( u )$ and because curves are smooth enough we choose as semi-metric the $L _ { 2 }$ -norm of the second derivatives of the curves, that is for $t _ { 1 } \neq t _ { 2 }$ ,

$$
\mathrm { d } ( X _ { t _ { 1 } } , X _ { t _ { 2 } } ) = \left( \int _ { - 1 } ^ { 1 } [ X _ { t _ { 1 } } ^ { ( 2 ) } ( s ) - X _ { t _ { 2 } } ^ { ( 2 ) } ( s ) ] ^ { 2 } \mathrm { d } s \right) ^ { 1 / 2 } .
$$

Summary statistics of $( \mathsf { S E } ^ { j } ) _ { j = 1 , \ldots , 5 0 0 }$ for discrete- and continuous-time estimators of the regresTable 1.sion function.   

<table><tr><td></td><td></td><td colspan="3">Continuous (x10-2)</td><td colspan="3">Discrete (x10-2)</td></tr><tr><td>MAR rate</td><td>T</td><td>50</td><td>200</td><td>1000</td><td>50</td><td>200</td><td>1000</td></tr><tr><td>p = 20%</td><td>Q25%</td><td>0.56</td><td>0.54</td><td>0.18</td><td>1.6</td><td>1.3</td><td>0.7</td></tr><tr><td></td><td>Median</td><td>2.4</td><td>2.58</td><td>1.25</td><td>7.5</td><td>3.7</td><td>2.1</td></tr><tr><td></td><td>Mean</td><td>5.39</td><td>4.62</td><td>4</td><td>15.11</td><td>11.8</td><td>9.1</td></tr><tr><td></td><td>Q75%</td><td>6.21</td><td>7.13</td><td>4.7</td><td>16.06</td><td>10.9</td><td>4.7</td></tr><tr><td>p = 40%</td><td>Q25%</td><td>0.49</td><td>0.6</td><td>0.3</td><td>1.7</td><td>1</td><td>0.5</td></tr><tr><td></td><td>Median</td><td>4.77</td><td>2.7</td><td>2.2</td><td>4.9</td><td>6.1</td><td>2</td></tr><tr><td></td><td>Mean</td><td>6.93</td><td>8.8</td><td>4.6</td><td>12.4</td><td>9.9</td><td>8.3</td></tr><tr><td></td><td>Q75%</td><td>9.39</td><td>10.6</td><td>4.5</td><td>16.05</td><td>11.3</td><td>11.1</td></tr></table>

We used the local cross-validation method on the $\kappa$ -nearest neighbours introduced in Ferraty and Vieu (2006) page 116 to select the optimal bandwidth for both discrete- and continuous-time regression estimators. The accuracy of the discrete- and continuous-time regression estimators is evaluated over $M = 5 0 0$ replications. The accuracy is measured, at each replication $j = 1 , \ldots , M ,$ , by using the squared errors $\mathrm { S E } _ { T } ^ { j } : = ( \widehat { m } _ { T } ^ { j } ( x ) - m ( x ) ) ^ { 2 }$ and $\operatorname { S E } _ { n } ^ { j } : = ( { \widehat { m } } _ { n } ^ { j } ( x ) - m ( x ) ) ^ { 2 }$ for the continuous-time and discrete-time estimators, respectively. Observe that the discrete-time estimator of the regression operator is defined as

$$
\widehat { m } _ { n } ( x ) : = \frac { \sum _ { t = 1 } ^ { n } \zeta _ { t } Y _ { t } \Delta _ { t } ( x ) } { \sum _ { t = 1 } ^ { n } \zeta _ { t } \Delta _ { t } ( x ) } .
$$

To get a better idea about the variability of the errors, Table 1 summarises the distribution of the squared errors (multiplied by $1 0 ^ { - 2 }$ ) $( \mathrm { S E } _ { T } ^ { j } ) _ { j = 1 , \dots , M }$ and $( \mathrm { S E } _ { n } ^ { j } ) _ { j = 1 , \dots , M }$ . It shows that continuous-time regression estimator is more accurate than the discrete-time one. Moreover, when $T$ increases the squared errors decrease faster when working with the continuous-time process.

# 4.2. Simulation 2: optimal sampling mesh selection

The purpose of this simulation is to investigate another aspect related to continuous-time processes. The selection of the ‘optimal’ sampling mesh is one of the most important topics in continuous-time processes.

First of all, we generate a continuous-time functional data process according to the following equation:

$$
X _ { t } ( s ) = Z _ { t } \left( 1 - \sin ( s - \pi / 3 ) \right) , \quad s \in [ 0 , \pi / 3 ] \quad \mathrm { a n d } \quad t \in [ 0 , T ] ,
$$

where $Z _ { t }$ is an OU process solution of the stochastic differential equation (19) and practically observed at the instants $t = \delta , 2 \delta , \ldots , n \delta$ with $n = 2 0 0$ fixed. Here, we take different values of sampling mesh $\delta$ , calculate the corresponding empirical version of the Mean Integrated Square Error $( \mathrm { M I S E } ( \delta ) )$ and identify the optimal mesh, say $\delta ^ { \star }$ , that minimises $\mathrm { M I S E } ( \delta )$ . Note that each curve observed at the instant $t$ is discretised at 100 equidistant points over the interval $[ 0 , \pi / 3 ]$ . The response variable is obtained following the hereafter nonlinear functional regression model (20), where the operator $m ( \cdot )$ is defined as

$$
m ( X _ { t } ) = \left( \int _ { 0 } ^ { \pi / 3 } X _ { t } ^ { \prime } ( s ) \mathrm { d } s \right) ^ { 2 } \quad \mathrm { a n d } \quad \varepsilon _ { t } \sim N ( 0 , 0 . 0 7 5 ) .
$$

Moreover, the missing at random mechanism in this simulation is also supposed to be the same as described in the first simulation as per Equation (21). For the tuning parameters used to build the estimator, we considered the quadratic kernel and given the shape of the true regression operator, which depends on the first derivative of the functional predictor, the Euclidean distance between the first-order derivatives of the curves is adopted as a semi-metric. Finally the bandwidth is selected according to the local cross-validation method based on the $\kappa$ -nearest neighbours as detailed in Ferraty and Vieu (2006, p. 116).

For each value of sampling mesh $\delta$ , the regression operator $m ( \cdot )$ is estimated over a grid of 50 different fixed curves and the whole procedure is repeated over $M = 5 0 0$ replications. Finally, the empirical MISE is calculated, for each sampling mesh $\delta$ , according to the following equation:

$$
\mathrm { M I S E } ( \delta ) : = \frac { 1 } { M } \sum _ { k = 1 } ^ { M } \frac { 1 } { 5 0 } \sum _ { j = 1 } ^ { 5 0 } \left( m ( x _ { j } ) - \widehat { m } _ { n , \delta } ^ { k } ( x _ { j } ) \right) ^ { 2 } .
$$

Observe that $\widehat { m } _ { n , \delta } ^ { k } ( \cdot )$ is the estimator of $m ( \cdot )$ , obtained at the $k$ th iteration, depends on the sampling mesh $\delta$ , so is the MISE.

Figure 2 displays the values of $\mathrm { M S E } ( \delta )$ obtained for different values of sampling mesh $\delta$ and missing at random rate of $1 0 \%$ , $5 0 \%$ and $0 \%$ (complete data), respectively. One can observe that higher is the missing at random rate, higher are the errors in estimating the regression operator.

Table 2 reports the optimal sampling mesh $\delta ^ { \star }$ , which minimises $\mathrm { M S E } ( \delta )$ , for different missing at random rates. It also provides some summary statistics to have an idea about the distribution of $\mathrm { M S E } ( \delta )$ for several values of $\delta$ . One can observe, from Table 2, that higher is the missing at random rate longer we need to observe the underlying process to collect the $n = 2 0 0$ observations to be able to reasonably estimate the regression operator. Indeed, when the data is complete the optimal time interval $T ^ { \star } = n \delta ^ { \star } = 2 0 0 \times 0 . 3 = 6 0 .$ However, when $M A R = 1 0 \%$ (resp. $5 0 \%$ ) the optimal time interval is equal to $T ^ { \star } = 2 0 0 \times 0 . 3 6 = 7 2$ (resp. $T ^ { \star } = 2 0 0 \times 0 . 3 8 = 7 6$ ). Consequently, it can be concluded that when the missing at random mechanism is heavily affecting the response variable, we need to collect data over a longer period of time. This allows to get sufficient information about the dynamic of the underlying continuous-time process and therefore get a better estimate of the regression operator.

![](images/4ced3b931d8c45dc6714a03f4f0b6d0cdcd5307e76c5176e0e5e809816e8ebe1.jpg)  
The MISE $( \delta )$ obtained for different values of sampling mesh $\delta$ and several missing at random Figurrates.

The optimal sampling mesh $( \delta ^ { \star } )$ obtained for different MAR rates and some summary statistics Tableof the ${ \mathsf { M I S E } } ( \delta )$ .   

<table><tr><td></td><td>8</td><td>MISE(8*)</td><td>MISE(8)</td><td>Q25%</td><td>Q50%</td><td>Q75%</td></tr><tr><td>MAR =0%</td><td>0.30</td><td>0.0476</td><td>0.0562</td><td>0.0510</td><td>0.0524</td><td>0.0550</td></tr><tr><td>MAR=10%</td><td>0.36</td><td>0.0555</td><td>0.0643</td><td>0.0587</td><td>0.0612</td><td>0.0635</td></tr><tr><td>MAR =50%</td><td>0.38</td><td>0.0635</td><td>0.0767</td><td>0.0714</td><td>0.0750</td><td>0.0770</td></tr></table>

# 4.3. Simulation 3: asymptotic confidence intervals

In this section, we are interested in evaluating the coverage rate, as well as the length, of the asymptotic confidence intervals given in (16). The effect of the sampling mesh on the coverage rate will also be discussed numerically. Since this paper aims to extend results in Delsol (2009) about confidence intervals to continuous-time functional data, we consider the same simulation framework.

Let $X _ { t } ( s ) = \cos ( Z _ { t } + \pi ( 2 s - 1 ) )$ , $s \in [ 0 , 1 ]$ and $t \in [ 0 , T ]$ , where $Z _ { t }$ is an OU process solution of the stochastic differential equation (19) observed at the instants $t =$ $\delta , 2 \delta , \ldots , n \delta$ with $n = 1 0 0 , 2 0 0$ fixed. Here, for comparison purpose, we consider two sampling mesh $\delta = 0 . 1 , 0 . 7$ . The regression operator is defined as $\begin{array} { r } { \overline { { m } } ( x ) = \frac { 1 } { 2 \pi } \int _ { 1 / 2 } ^ { 3 / 4 } ( x ^ { \prime } ( s ) ) ^ { 2 } \mathrm { d } s , } \end{array}$ while the errors $\left\{ \varepsilon _ { t } \right\}$ are independent centred normal random variable with variance $0 . 1 \times s _ { n } ^ { 2 }$ , where $s _ { n } ^ { 2 }$ is the empirical variance of $\{ m ( X _ { 1 } ) , \ldots , m ( X _ { n } ) \}$ . Because the regression operator is defined as a function of the derivative of the functional random variable, the appropriate semimetric to be used in such case is based on the first derivative of the curve (see (22)). Moreover, the quadratic kernel is used to perform this simulation. The optimal bandwidth is selected based on local cross-validation method on the $\kappa$ -nearest neighbours. The missing at random rate is simulated according the conditional probability distribution given in (21).

For a fixed $\alpha \in ( 0 , 1 )$ , the asymptotic $( 1 - \alpha )$ -confidence intervals for $m ( x )$ with $x \in \Xi$ are computed and compared for several values of sample size $n$ and sampling mesh $\delta .$ Here $\Xi : = \{ x _ { 1 } , \ldots , x _ { n _ { \Xi } } \}$ is a grid of $n _ { \Xi } = 5 0$ independently simulated curves where the regression operator is estimated. For every fixed curve $x \in \Xi$ , a number of $M = 5 0 0$ replications is considered to approximate the coverage rate. In this simulation $1 - \alpha = 0 . 9 5 , 0 . 9 0$ were considered.

As expected, Table 3 shows that the average coverage rate varies with the sample size $n$ , the sampling mesh and the MAR Rate. Higher are the sample size and the sampling mesh and smaller is the MAR rate closer will be the average coverage rate to $1 - a$ . Moreover, one can also observe that the asymptotic confidence intervals length decreases when the sample size increases and the MAR rate decreases.

Figure 3 (resp. Figure 4) displays an example of asymptotic confidence intervals obtained for the 50 curves in the testing sample when $n = 1 0 0$ , the MAR rate $= 0 \%$ , $2 5 \%$ , $4 5 \%$ $\delta = 0 . 7$ and $1 - \alpha = 0 . 9 5$ (resp. $1 - \alpha = 0 . 9$ ). One can observe that the coverage rate decreases with an increase in the MAR rate. Similar results are also obtained when $\delta = 0 . 1$ .

Average coverage over the grid  and average confidence interval length appears in brackets.   

<table><tr><td></td><td></td><td colspan="2">0.95</td><td colspan="2">0.90</td></tr><tr><td>1-α</td><td>MAR rate</td><td>n = 100</td><td>n= 200</td><td>n = 100</td><td>n = 200</td></tr><tr><td>δ=0.1</td><td>MAR=0%</td><td>0.762 (0.148)</td><td>0.778 (0.104)</td><td>0.743 (0.126)</td><td>0.737 (0.088)</td></tr><tr><td></td><td>MAR =25%</td><td>0.750 (0.149)</td><td>0.767 (0.104)</td><td>0.725 (0.125)</td><td>0.731 (0.088)</td></tr><tr><td></td><td>MAR= 45%</td><td>0.722 (0.152)</td><td>0.750 (0.106)</td><td>0.692 (0.127)</td><td>0.697 (0.089)</td></tr><tr><td>δ=0.7</td><td>MAR=0%</td><td>0.817 (0.179)</td><td>0.851 (0.148)</td><td>0.850 (0.144)</td><td>0.81(0.124)</td></tr><tr><td></td><td>MAR=25%</td><td>0.811 (0.178)</td><td>0.841(0.149)</td><td>0.831(0.144)</td><td>0.798 (0.123)</td></tr><tr><td></td><td>MAR=45%</td><td>0.766 (0.179)</td><td>0.791 (0.148)</td><td>0.771 (0.145)</td><td>0.738 (0.124)</td></tr></table>

![](images/32b9dd681908780c408dc9a166f50b301f11e58267e4c9ed9f9733546c3ef3fd.jpg)  
Asymptotic $9 5 \%$ confidence intervals when $\delta = 0 . 7$ and $n = 1 0 0 .$ Red dot represents the true Figure 3.regression function value at any fixed curve $x \in \Xi$ and the black dot is its estimation. The vertical lines represent the confidence intervals.

![](images/f4e51fe0463c44abc2e489c9096c4654f0d23a4d0f83dc2abeb2616ad495747f.jpg)  
Asymptotic $9 0 \%$ confidence intervals when $\delta = 0 . 7$ and $n = 1 0 0 .$ Red dot represents the true Figure 4.regression function value at any fixed curve $x \in \Xi$ and the black dot is its estimation. The vertical lines represent the confidence intervals.

# 5. Applications to real data

# 5.1. Application 1: prediction of financial asset returns

In Financial market, despite the modern technology, which allows to collect data at a very fine time scale, financial data can still be missing. For instance, there are some regular holidays, such as Thanksgiving Day and Christmas, for which stock price data are missing. There are many other technical reasons (such as breakdown in devises recording data, computers’ sudden shutdowns, . . . ) that make stretches of data missing.

This section aims to assess the performance of the estimator proposed in this paper on missing at random financial functional time series. The International Business Machine cooperation (IBM) asset price is considered as the response variable and the Standard & Poor’s 500 (SP500) stock market index as predictor. While the IBM asset price is observed at a daily frequency from 24 March 2016 to 28 September 2016, the SP500 is observed every minute during the same period. Note that the daily trading activity lasts for 7 hours excluding the weekend. Since in this paper, we are interested in stationary processes, a first-order differentiation of the IBM daily asset price and the SP500 stock market index was considered to make the original time series stationary.

Our sample here can be denoted as follows: $( X _ { d } , Y _ { d } ) _ { d = 1 , . . . 1 2 9 }$ , where the sample size $n = 1 2 9$ is the total number of trading days from 24 March 2016 to 28 September 2016 after first differentiation of the original time series, $Y _ { d } = \Delta \mathrm { I B M } _ { d } , X _ { d } = \{ X _ { d } ( s ) : = \Delta \mathrm { S P } 5 0 0 _ { d } ( s ) :$ $1 \leq s \leq 4 2 0 ]$ . Originally, the data are completely observed. Therefore, to validate our estimator, we artificially create missing observations. We assume here that the missing data are generated according to the conditional probability distribution given in (21). We split the original sample into training and testing subsets. Our purpose then is to predict the IBM asset price in the testing subset using the regression operator. Three MAR rates $0 \%$ (complete data), $2 0 \%$ and $4 5 \%$ were considered to test the performance of the estimator in terms of prediction. Similarly, as in the simulation section, we considered here the quadratic kernel and the bandwidth was selected using the cross-validation method on the $\kappa$ -nearest neighbours. For the semi-metric, because the curves are not smooth (as it can be seen in Figure 5, right panel) we use the PCA-semi-metric, say $d _ { 4 } ^ { \mathrm { P C A } } ( \cdot , \cdot )$ , based on the projection on the four eigenfunctions, $\nu _ { 1 } ( \cdot ) , \ldots , \nu _ { 4 } ( \cdot )$ , associated to the four largest eigenvalues of the empirical covariance operator of the functional predictor $X$ :

$$
d _ { 4 } ^ { \mathrm { P C A } } ( X _ { t _ { 1 } } , X _ { t _ { 2 } } ) = \sqrt { \sum _ { k = 1 } ^ { 4 } \left( \int _ { 0 } ^ { 4 2 0 } \left( X _ { t _ { 1 } } ( s ) - X _ { t _ { 2 } } ( s ) \right) \nu _ { k } ( s ) \mathrm { d } s \right) ^ { 2 } } .
$$

As criteria to measure the accuracy of the estimator in predicting 30 observations in the testing subset, we considered the Absolute Error: $\mathrm { A E } _ { t } : = | Y _ { t } - m _ { n } ( X _ { t } ) |$ for $t = 1 , \ldots , 3 0$ . Figure 6 displays the distribution of the absolute errors obtained when MAR rate is $0 \%$ , $2 0 \%$ and $4 5 \%$ , respectively. One can clearly observe the effect of the MAR rate on the quality of the prediction. Higher is the MAR rate lower is the quality of prediction.

Moreover, we build a $9 5 \%$ prediction interval for the IBM asset price in the testing subset. Figure 7 shows that the coverage rate is sensitive to the percentage MAR data in the training subset.

![](images/03c493806cf59fcd3f0b42a0f315c01a6d5781e93d721c51c424592a8445be3d.jpg)  
Left: First-order differentiated IBM asset price. Right: First-order differentiated SP500 intraday Figure 5.(minute frequency) stock market index curves.

![](images/ebe2151d3a59bc4bfd77d956ca2857dd3688f0a9d995e13265cb77d30a0d50dd.jpg)  
The $( \mathsf { A E } ) _ { t = 1 , \ldots , 3 0 }$ obtained for different values of missing at random rates.

![](images/dc3ffd72be6d58e42333e486983ac80a6f2c94948ea056254a7cc5584155cd23.jpg)  
Figure 7. A $9 5 \%$ prediction interval of the IBM asset price in the testing subset. Red dot represents the Figure 7.true values of IBM asset price and the black dot is their prediction using the regression operator. The vertical lines represent the prediction intervals.

# 5.2. Application 2: Daily peak electricity demand imputation

By accurately predicting household peak load, utility companies can better balance the overall electricity demand and supply. This information helps in optimising power generation and distribution, ensuring a stable and reliable electricity grid. It also allows to plan for peak demand periods and avoid potential blackouts or overloading of the grid. Moreover, predicting peak loads empowers consumers with information about their electricity consumption patterns. Thus, with this knowledge, households can make informed decisions to manage their energy usage more effectively, reduce electricity bills and contribute to energy conservation efforts. Furthermore, peak load predictions enable demand response programs, where utility companies offer incentives for consumers to adjust their energy consumption during peak periods, thereby reducing strain on the grid.

For these reasons, electricity companies deployed smart meters to replace the mechanical one. This new generation of smart meters allows to record the electricity demand of any household at very fine time scale and send it to the information system. The transmission of the information from the smart meter towards the information system goes usually through WIFI or optical fibre networks which are significantly dependent on the weather conditions, among several other factors. Therefore, the calculation of the daily peak electricity demand might be subject to missing at random mechanism due to bad weather conditions.

Figure 8 displays the daily peak load $\{ Y _ { t _ { k } } \} _ { k = 1 , \ldots , 1 0 0 9 }$ obtained from a household smart meter from 24 September 1996 to 29 June 1999 (leading to a total of $n = 1 0 0 9$ days). The original data contains $1 0 \%$ of missing observations. Here, we assume that the intraday (3-hour frequency) temperature curve $\{ X _ { t _ { k } } ( s ) : s = 3 , 6 , . . . , 2 4 \}$ explains the missingness mechanism in the daily peak demand. Figure 9 displays the intraday, 3-hour frequency, temperature curves. Our purpose in this application is to impute the missing data in the peak demand process using the initial estimator of the regression operator $\widehat { m } _ { n } ( x ) : =$ $\widehat { m } _ { \psi , n } ( x , \gamma )$ defined in (17) with $\psi ( Y _ { t _ { k } } , y ) = Y _ { t _ { k } }$ . Figure 10 displays the imputed peak electricity demand process obtained according to the following formula:

$$
\widetilde { Y } _ { t _ { k } } = \delta _ { t _ { k } } Y _ { t _ { k } } + ( 1 - \delta _ { t _ { k } } ) \widehat { m } _ { n } ( X _ { t _ { k } } ) .
$$

If $Y _ { t _ { k } }$ is observed (that is $\delta _ { t _ { k } } = 1$ ), then $\widetilde { Y } _ { t _ { k } } = Y _ { t _ { k } }$ , otherwise $Y _ { t _ { k } }$ is missing (i.e. $\delta _ { t _ { k } } = 0$ ) and will be imputed by $\widehat { m } _ { n } ( X _ { t _ { k } } )$ . The red dots in Figure 10 represent the imputed values of the missing observations in the peak electricity demand process. Figure 11 shows the $9 5 \%$ confidence intervals around the missing values of the peak load.

![](images/fc8d9f58d654383d64c3ed9fb68e534823ff70cd5647944cbc307b9eea626cc0.jpg)  
Daily peak electricity demand of a household containing $1 0 \%$ missing data.

![](images/b075f9b4f84d634c96d423be51fe32a1c03f05981611da9d2acff59e40238c68.jpg)  
Intraday temperature curves. Coloured curves are for the missing peak load days.

![](images/aa165071a0dd0d465f1dacd1bea2eff5bb041c22e155ad0fb8b269d5f1a9c101.jpg)  
Daily peak electricity demand process. Red dots represent values of imputed missing data.

![](images/3552166b0eab4e99227098297eb897ae633b5ef710f82d89428e481bea04a9b0.jpg)  
Figure 11. $9 5 \%$ confidence intervals for the imputed values.

# 6. Discussion of a special case: conditional quantiles

Let $x \in { \mathcal { E } }$ be fixed and $y \in \mathbb { R }$ , then if $\psi ( Y , y ) = \mathbb { 1 } _ { \{ \mathrm { l } - \infty , y \} \mathrm { l } } ( Y )$ the operator $m _ { \psi } ( x , y )$ is the conditional cumulative distribution function (df) of $Y$ given $X = x ,$ namely $F ( y | x ) =$ $\mathbb { P } ( Y \leq y | X = x )$ which may be estimated by ${ \widehat { F } } _ { T } ( y \mid x ) : = { \widehat { m } } _ { \psi , T } ( x , y )$ . For a given $\alpha \in$ $( 0 , 1 )$ , the αth -order conditional quantile of the distribution of $Y$ given $X = x$ is defined as $q _ { \alpha } ( x ) = \operatorname* { i n f } \{ y \in \mathbb { R } : F ( y \mid x ) \geq \alpha \}$ .

Notice that, whenever $F ( \cdot \mid x )$ is strictly increasing and continuous in a neighbourhood of $q _ { \alpha } ( x )$ , the function $F ( \cdot \mid x )$ has a unique quantile of order $\alpha$ at a point $q _ { \alpha } ( x )$ , that is $F ( q _ { \alpha } ( x ) \mid x ) = \alpha .$ . In such case

$$
q _ { \alpha } ( { \boldsymbol x } ) = { \boldsymbol F } ^ { - 1 } ( \alpha \mid { \boldsymbol x } ) = \operatorname* { i n f } \{ { \boldsymbol y } \in \mathbb { R } : F ( { \boldsymbol y } \mid { \boldsymbol x } ) \geq \alpha \} ,
$$

which may be estimated uniquely by $\widehat { q } _ { T , \alpha } ( x ) = \widehat { F } _ { T } ^ { - 1 } ( \alpha | x )$ . Conditional quantiles have been widely studied in the literature when the predictor $X$ is of finite dimension, see for instance, Gannoun et al. (2003) and Ferraty et al. (2005) for dependent functional data.

(a) Almost sure pointwise and uniform convergence

Under the same conditions of Theorem 3.1, the statement (9) still holds for the estimator of the cumulative conditional distribution function $\widehat { F } _ { T } ( y \vert x )$ . That is ${ \widehat { F } } _ { T } ( \alpha \mid x )$ converges, almost surely, towards $F ( \boldsymbol { y } \vert \boldsymbol { x } )$ with a rate $\mathcal { O } ( h _ { T } ^ { \beta } ) + \mathcal { O } ( \sqrt { \log T / ( T \phi ( h _ { T } ) ) } )$ .

Consequently, since $F ( q _ { \alpha } ( x ) \mid x ) = \alpha = { \widehat { F } } _ { T } { \widehat { ( q _ { T , \alpha } ( x ) \mid x ) } }$ and $\widehat { F } _ { T } ( \cdot | x )$ is continuous and strictly increasing, then we have $\forall \epsilon > 0$ , $\exists \eta ( \epsilon ) > 0$ , $\forall y , | { \widehat { F } } _ { T } ( y | x ) - { \widehat { F } } _ { T } ( q _ { \alpha } ( x ) | x ) | \leq$ $\eta ( \epsilon ) \Rightarrow | y - q _ { \alpha } ( x ) | \leq \epsilon$ which implies that, $\forall \epsilon > 0$ , $\exists \eta ( \epsilon ) > 0$ ,

$$
\begin{array} { r l } & { \mathbb { P } \left( \left| \widehat { q } _ { T , \alpha } ( x ) - q _ { \alpha } ( x ) \right| \geq \eta ( \epsilon ) \right) \leq \mathbb { P } \left( \left| \widehat { F } _ { T } ( \widehat { q } _ { T , \alpha } ( x ) | x ) - \widehat { F } _ { T } ( q _ { \alpha } ( x ) | x ) \right| \geq \eta ( \epsilon ) \right) } \\ & { \qquad = \mathbb { P } \left( | F ( q _ { \alpha } ( x ) | x ) - \widehat { F } _ { T } ( q _ { \alpha } ( x ) | x ) \geq \eta ( \epsilon ) \right) . } \end{array}
$$

Therefore, the statement (9) still holds for the conditional quantile estimator $\widehat { q } _ { T , \alpha } ( x )$ whenever conditions of Theorem 3.1 are satisfied. Ferraty et al. (2005) derived similar pointwise convergence rate by inverting the estimator of the conditional cumulative distribution function. Their result has been obtained under mixing condition and additional assumptions on the joint distribution, and the Lipschitz condition on $F ( \boldsymbol { y } \vert \boldsymbol { x } )$ and its derivatives with respect to $\boldsymbol { y }$ .

Regarding the almost sure uniform convergence, observe that under conditions of Theorem 3.2, the statement (11) still holds true for the $\begin{array} { r } { \operatorname* { s u p } _ { y \in S } \operatorname* { s u p } _ { x \in \mathcal { C } } | \widehat { F } _ { T } ( y ( x ) - F _ { T } ( y ( x ) | , } \end{array}$ , when $\psi ( Y , y )$ is replaced by $\mathbb { 1 } _  \{ \} - \infty , y ] \} ( Y )$ . Moreover, assume that, for fixed $x _ { 0 } \in { \mathcal { C } }$ , $F ( \boldsymbol { y } \vert \boldsymbol { x } _ { 0 } )$ is differentiable at $q _ { \alpha } ( x _ { 0 } )$ with ${ \frac { \partial } { \partial y } } F ( y | x _ { 0 } ) | _ { y = q _ { \alpha } ( x _ { 0 } ) } : = g ( q _ { \alpha } ( x _ { 0 } ) | x _ { 0 } ) > \nu > 0 ,$ , where $\nu$ is a real number, and $g ( \cdot \mid x )$ is uniformly continuous for all $x \in { \mathcal { C } }$ . Knowing that $\widehat F _ { T } ( \widehat q _ { T , \alpha } ( x ) \mid x ) = F ( q _ { \alpha } ( x ) \mid x ) = \alpha$ and making use of a Taylor’s expansion of the function $F ( \widehat { q } _ { T , \alpha } ( x ) \mid x )$ around $q _ { \alpha } ( x )$ , we can write

$$
F ( \widehat { q } _ { T , \alpha } ( x ) \mid x ) - F ( q _ { \alpha } ( x ) \mid x ) = \left( \widehat { q } _ { T , \alpha } ( x ) - q _ { \alpha } ( x ) \right) g \left( q _ { T , \alpha } ^ { * } ( x ) \mid x \right)
$$

where $q _ { T , a } ^ { * } ( x )$ lies between $q _ { \alpha } ( x )$ and $\widehat { q } _ { T , \alpha } ( x )$ . It follows then from (24) that the inequality (23) still holds true uniformly in $x$ and $y$ . Moreover, the fact that $\widehat { q } _ { T , a } ( x )$ converges a.s. towards $q _ { \alpha } ( x )$ as $T$ goes to infinity, combined with the uniformly continuity of $g ( \cdot \mid x )$ ,

allow to write that

$$
\operatorname* { s u p } _ { x \in \mathcal { C } } \left| \widehat { q } _ { T , \alpha } ( x ) - q _ { \alpha } ( x ) \right| \operatorname* { s u p } _ { x \in \mathcal { C } } \left| g \left( q _ { \alpha } ( x ) \left| x \right. \right) \right| = O _ { a . s . } \left( \operatorname* { s u p } _ { y \in S } \operatorname* { s u p } _ { x \in \mathcal { C } } \left| \widehat { F } _ { T } ( y | x ) - F ( y | x ) \right| \right) .
$$

Since $g ( q _ { \alpha } ( x ) | x )$ is uniformly bounded from below, we can then claim that the estimator $\widehat { q } _ { T , \alpha } ( x )$ converges uniformly towards $q _ { \alpha } ( x )$ with the same convergence rate given in (11), as $T$ goes to infinity.

(b) Continuous-time confidence intervals

Confidence intervals for the conditional quantiles $q _ { \alpha } ( x )$ may be obtained according to the following steps. First, consider a Taylor’s expansion of $\widehat { F } _ { T } ( \cdot | x )$ around $q _ { \alpha } ( x )$ and making use of the fact that $\widehat { q } _ { T , a } ( x )$ converges a.s. towards $q _ { \alpha } ( x )$ as $T$ goes to infinity, one gets

$$
\widehat { \boldsymbol { q } } _ { T , \alpha } ( \boldsymbol { x } ) - \boldsymbol { q } _ { \alpha } ( \boldsymbol { x } ) = - \frac { 1 } { \widehat { \ g } _ { T } \left( \boldsymbol { q } _ { \alpha } ( \boldsymbol { x } ) \mid \boldsymbol { x } \right) } \left( \widehat { F } _ { T } ( \boldsymbol { q } _ { \alpha } ( \boldsymbol { x } ) \mid \boldsymbol { x } ) - F ( \boldsymbol { q } _ { \alpha } ( \boldsymbol { x } ) \mid \boldsymbol { x } ) \right) ,
$$

where ${ \widehat { g } } _ { T } ( \cdot \vert x )$ is a consistent estimator of $g ( \cdot \mid x )$ . Then, replacing $\psi ( Y , y )$ by the indicator function, we get under conditions of Corollary 3.7, the following $( 1 - \alpha )$ confidence intervals for $q _ { \alpha } ( x )$ :

$$
\widehat { q } _ { T , \alpha } ( x ) \ \pm c _ { 1 - \alpha / 2 } \frac { \sqrt { M _ { T , 2 } } } { M _ { T , 1 } \widehat { g } _ { T } \left( \widehat { q } _ { T , \alpha } ( x ) \vert x \right) } \sqrt { \frac { \alpha ( 1 - \alpha ) } { T F _ { x , T } ( h _ { T } ) p _ { T } ( x ) } } , \quad \mathrm { a s } \ T \to \infty .
$$

# Note

1. For any $0 \leq s < t \leq T$ such that $t - s \leq \alpha _ { 0 }$ , there exists a non-negative continuous random function $f _ { t , s } ( \omega , x )$ such that $\mathrm { s u p } _ { s \geq 0 }$ $\begin{array} { r } { \mathtt { p } _ { s \ge 0 , \omega \in \Omega } | f _ { t , s } ( \omega , x ) | \le b _ { s , \alpha _ { 0 } } ( x ) p . s } \end{array}$ . where $b _ { s , \alpha _ { 0 } } ( x )$ is a deterministic function.

# Acknowledgments

Open Access funding provided by the Qatar National Library.

We thank the editor, associate editor and the two referees for their valuable and constructive comments which helped improve the manuscript substantially.

# Disclosure statement

No potential conflict of interest was reported by the author(s).

# References

Andrews, D.W.K. (1984), ‘Non-strong Mixing Autoregressive Processes’, Journal of Applied Probability, 21, 930–934.   
Bosq, D. (1998), Nonparametric Statistics for Stochastic Processes: Estimation and Prediction, Lecture Notes in Statistics, Vol. 110 (2nd ed.), New York: Springer-Verlag.   
Bouzebda, S., and Didi, S. (2017), ‘Asymptotic Results in Additive Regression Model for Strictly and Ergodic Continuous Times Processes’, Communications in Statistics-Theory and Methods, 46(5), 2454–2493.   
Chaouch, M., and Laïb, N. (2019), ‘Optimal Asymptotic MSE of Kernel Regression Estimate for Continuous Time Processes with Missing At Random Response’, Statistics and Probability Letters, 154, 108532.   
Chaouch, M., and Laïb, N. (2023), ‘Supplement to “Regression estimation for continuous time functional data processes with missing at random response”’.   
Cheng, P.E. (1994), ‘Nonparametric Estimation of Mean Functionals with Data Missing at Random’, Journal of the American Statistical Association, 89, 81–87.   
Cheridito, P., Kawaguchi, H., and Maejima, M. (2003), ‘Fractional Ornstein–Uhlenbeck Processes’, Electronic Journal of Probability, 8(3), 1–14.   
Chesneau, C, and Maillot, B. (2014), ‘Superoptimal Rate of Convergence in Nonparametric Estimation for Functional Valued Processes’, International Scholarly Research Notices, 2014, 264217.   
Chu, C.K., and Cheng, P.E. (2003), ‘Nonparametric Regression Estimation with Missing Data’, Journal of Statistical Planning and Inference, 48, 85–99.   
de la Peña, V.H., and Giné, E. (1999), Decoupling: From Dependence to Independence, Probability and Its Applications, New York: Springer-Verlag.   
Delsol, L. (2009), ‘Advances on Asymptotic Normality in Non-parametric Functional Time Series Analysis’, Statistics, 43, 13–33.   
Didi, S., and Louani, D. (2014), ‘Asymptotic Results for the Regression Function Estimate on Continuous Time Stationary Ergodic Data’, Journal Statistics & Risk Modeling, 31(2), 129–150.   
Doukhan, P. (2018), Stochastic Models for Time Series, New York: Springer.   
Doukhan, P., and Louhichi, S. (1999), ‘A New Weak Dependence Condition and Applications to Moment Inequalities’, Stochastic Processes and Their Applications, 84, 313–342.   
Efromovich, S. (2011), ‘Nonparametric Regression with Responses Missing at Random’, Journal of Statistical Planning and Inference, 141, 3744–3752.   
Ferraty, F., Laksaci, A., Tadj, A., and Vieu, P. (2010), ‘Rate of Uniform Consistency for Nonparametric Estimates with Functional Variables’, Journal of Statistical Planning and Inference, 140, 335–352.   
Ferraty, F., Mas, A., and Vieu, P. (2007), ‘Nonparametric Regression on Functional Data: Inference and Practical Aspects’, Australian & New Zealand Journal of Statistics, 49(3), 267–286.   
Ferraty, F., Rabhi, A., and Vieu, P. (2005), ‘Special Issue on Quantile Regression and Related Methods’, Sankhyà : The Indian Journal of Statistics, 67(2), 378–398.   
Ferraty, F., Sued, M., and Vieu, P. (2013), ‘Mean Estimation with Data Missing at Random for Functional Covariables’, Statistics, 47(4), 688–706.   
Ferraty, F, and Vieu, P. (2006), Nonparametric Modelling for Functional Data, Methods, Theory, Applications and Implementations, London: Springer-Verlag.   
Francq, C., and Zakoïan, J.M. (2010), GARCH Models: Structure, Statistical Inference and Financial Applications, John Wiley and Sons Ltd.   
Gannoun, A., Saracco, J., and Yu, K. (2003), ‘Nonparametric Prediction by Conditional Median and Quantiles’, Journal of Statistical Planning and Inference, 117, 207–223.   
Giraitis, L., and Leipus, R. (1995), ‘A Generalized Fractionally Differencing Approach in Longmemory Modeling’, Lithuanian Mathematical Journal, 35(1), 53–65.   
González-Manteiga, W., and Pérez-González, A. (2004), ‘Nonparametric Mean Estimation with Missing Data’, Communications in Statistics-Theory and Methods, 33(2), 277–303.   
Guégan, D., and Ladoucette, S. (2001), ‘Non-mixing Properties of Long Memory Processes’, Comptes Rendus De L’Académie Des Sciences – Series I – Mathematics, 333(1), 373–376.   
Hall, P., and Heyde, C. (1980), Martingale Limit Theory and Its Application, New York: Academic Press.   
Laïb, N., and Louani, D. (2010), ‘Nonparametric Kernel Regression Estimation for Functional Stationary Ergodic Data: Asymptotic Properties’, Journal of Multivariate Analysis, 101(10), 2266–2281.   
Laïb, N., and Louani, D. (2011), ‘Rates of Strong Consistencies of the Regression Function Estimator for Functional Stationary Ergodic Data’, Journal of Statistical Planning and Inference, 141(1), 359–372.   
Liang, H., Wang, S., and Carroll, R.J. (2007), ‘Partially Linear Models with Missing Response Variables and Error-prone Covariates’, Biometrika, 94(1), 185–198.   
Ling, N., Liang, L., and Vieu, P. (2015), ‘Nonparametric Regression Estimation for Functional Stationary Ergodic Data with Missing At Random’, Journal of Statistical Planning and Inference, 162, 75–87.   
Little, R.J.A., and Rubin, D.B. (2002), Statistical Analysis with Missing Data (2nd ed.), New York: John Wiley.   
Maillot, B. (2008), ‘Propriétś Asymptotiques de Quelques Estimateurs Non-paramétriques Pour des Variables Vectorielles et Fonctionnelles’, Thése de Doctorat de l’Université Paris 6.   
Maslowski, B., and Pospíšil, P. (2008), ‘Ergodicity and Parameter Estimates for Infinite-dimensional Fractional Ornstein–Uhlenbeck Process’, Applied Mathematics and Optimization, 57, 401–429.   
Nittner, T. (2003), ‘Missing At Random (MAR) in Nonparametric Regression, a Simulation Experiment’, Statistical Methods and Applications, 12, 195–210.   
Sikov, A. (2018), ‘A Brief Review of Approaches to Non-ignorable Non-response’, International Statistical Review, 86, 415–441.   
Tsiatis, A. (2006), Semiparametric Theory and Missing Data, New York: Springer.

# Appendix. Proofs of main results

In this section and for sake of simplification will denote $\widehat { m } _ { T } ( x , y )$ and $m ( x , y )$ for $\widehat { m } _ { \psi , T } ( x , y )$ and $m _ { \psi } ( x , y )$ , respectively, and $\psi _ { y } ( Y _ { t } )$ for $\psi ( Y _ { t } , y )$ . Consider now the following quantities:

$$
\begin{array} { r l } & { \boldsymbol { Q } _ { T } ( x , y ) : = ( \widehat { m } _ { T , 2 } ( x , y ) - \overline { { m } } _ { T , 2 } ( x , y ) ) - m ( x , y ) ( \widehat { m } _ { T , 1 } ( x ) - \overline { { m } } _ { T , 1 } ( x ) ) , } \\ & { \boldsymbol { R } _ { T } ( x , y ) : = - B _ { T } ( x , y ) ( \widehat { m } _ { T , 1 } ( x ) - \overline { { m } } _ { T , 1 } ( x ) ) . } \end{array}
$$

We have then

$$
\widehat { m } _ { T } ( x , y ) - m ( x , y ) = \widehat { m } _ { T } ( x , y ) - C _ { T } ( x , y ) + B _ { T } ( x , y ) = B _ { T } ( x , y ) + \frac { Q _ { T } ( x , y ) + R _ { T } ( x , y ) } { \widehat { m } _ { T , 1 } ( x ) } .
$$

We start first by stating some technical lemmas that will be used later.

Lemma A.1: Assume that assumptions (A1)–(A2) are satisfied, then we have for any $j \geq 1$ and $\ell \geq 1$

$$
\begin{array} { r l } & { \frac { 1 } { \phi ( h _ { T } ) } \mathbb { E } ( \Delta _ { t } ^ { \ell } ( x ) \mid \mathcal { F } _ { j - 2 } ) = M _ { \ell } f _ { t , T _ { j - 2 } } ( x ) + \mathcal { O } _ { a . s . } ( \frac { g _ { t , T _ { j - 2 } , x } ( h _ { T } ) } { \phi ( h _ { T } ) } ) , } \\ & { \frac { 1 } { \phi ( h _ { T } ) } \mathbb { E } ( \Delta _ { t } ^ { \ell } ( x ) ) = M _ { \ell } f ( x ) + o ( 1 ) . } \end{array}
$$

Proof: The proof is similar to the proof of Lemma 1 of Laïb and Louani (2010).

Lemma A.2: Let $( Z _ { n } ) _ { n \geq 1 }$ be a sequence of real martingale differences with respect to the sequence of $\sigma$ -fields $( { \mathcal { F } } _ { n } = \sigma ( Z _ { 1 } , . . . Z _ { n } ) ) _ { n \geq 1 }$ where $\sigma ( Z _ { 1 } , \ldots , Z _ { n } )$ is the sigma-field generated by the random variables $Z _ { 1 } , \ldots , Z _ { n }$ . Set $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ . For any $p \geq 2$ and any $n \geq 1$ , assume that there exist some nonnegative constants $C$ and $d _ { n }$ such that $\mathbb { E } ( Z _ { n } ^ { p } | \mathcal { F } _ { n - 1 } ) \le C ^ { p - 2 } p ! d _ { n } ^ { 2 }$ almost surely. Then, for any $\epsilon >$ 0, we have

$$
\mathbb { P } \left( | S _ { n } | > \epsilon \right) \le 2 \exp \left\{ - \frac { \epsilon ^ { 2 } } { 2 ( D _ { n } + C \epsilon ) } \right\} ,
$$

where $\textstyle D _ { n } = \sum _ { i = 1 } ^ { n } d _ { i } ^ { 2 }$

Proof: See Theorem 8.2.2 of de la Peña and Giné (1999).

Proof of Theorem 3.4: From (A3) and Lemma 1.2 (in Chaouch and Laïb 2023), we have for $T$ large enough

$$
\begin{array} { l } { \displaystyle \mathrm { M S E } ( x , y ) = \mathbb { E } \left( \widehat { m } _ { T } ( x , y ) - m ( x , y ) \right) ^ { 2 } \simeq \mathbb { E } \left( B _ { T } ( x , y ) + \frac { Q _ { T } ( x , y ) + R _ { T } ( x , y ) } { p ( x ) } \right) ^ { 2 } } \\ { \displaystyle \simeq \mathbb { E } \left( B _ { T } ^ { 2 } ( x , y ) \right) + \frac { 1 } { p ^ { 2 } ( x ) } \left[ \mathbb { E } \left( Q _ { T } ^ { 2 } ( x , y ) \right) + \mathbb { E } \left( R _ { T } ^ { 2 } ( x , y ) \right) \right] , } \end{array}
$$

where the products $2 \mathbb { E } [ B _ { T } ( x , y ) ( Q _ { T } ( x , y ) + R _ { T } ( x , y ) ) ]$ and $2 \mathbb { E } [ Q _ { T } ( x , y ) \times R _ { T } ( x , y ) ]$ have been ignored because by the Cauchz–Schwarz inequality

$$
\begin{array} { r l } & { \mathbb { E } [ B _ { T } ( x , y ) ( Q _ { T } ( x ) + R _ { T } ( x , y ) ) ] \leq \mathbb { E } ( B _ { T } ^ { 2 } ( x , y ) ) ^ { 1 / 2 } \times \mathbb { E } ( ( Q _ { T } ( x , y ) + R _ { T } ( x , y ) ) ^ { 1 / 2 }  } \\ & { \qquad \leq \operatorname* { m a x } \{ \mathbb { E } ( B _ { T } ^ { 2 } ( x , y ) ) , \mathbb { E } [ ( Q _ { T } ( x , y ) + R _ { T } ( x , y ) ) ^ { 2 } ] \} . } \end{array}
$$

We have the same inequality for the second product. The proof of Theorem 3.4 results from Proposition 3.3 and Lemma A.3 below, which gives an upper bound of the expectation of $Q _ { T } ^ { 2 } ( x , y )$ and $R _ { T } ^ { 2 } ( x , y )$ , respectively.

Lemma A.3: Assume that (A1)–(A3) hold true, then we have

$$
\mathbb { E } ( Q _ { T } ^ { 2 } ( x , y ) ) \simeq \frac { 4 p ( x ) ( W _ { 2 } ( x , y ) + ( m ( x , y ) ) ^ { 2 } ) M _ { 2 } } { T \phi ( h _ { T } ) M _ { 1 } ^ { 2 } f ( x ) } .
$$

Proof: Ignoring the product term as above, one may write

$$
\begin{array} { r l } & { \mathbb { E } \left[ Q _ { T } ^ { 2 } ( x , y ) \right] \simeq \mathbb { E } \left[ \widehat { m } _ { T , 2 } ( x , y ) - \overline { { m } } _ { T , 2 } ( x , y ) \right] ^ { 2 } } \\ & { \quad \quad \quad \quad \quad + m ^ { 2 } ( x , y ) \mathbb { E } \left[ \widehat { m } _ { T , 1 } ( x ) - \overline { { m } } _ { T , 1 } ( x ) \right] ^ { 2 } : = I _ { T , 1 } + m ^ { 2 } ( x , y ) I _ { T , 2 } . } \end{array}
$$

The terms $I _ { T , 1 }$ and $I _ { T , 2 }$ can be handled similarly. Let us now evaluate the first one. Since $( T _ { j } =$ $j \delta ) _ { 0 \geq j \geq n }$ is a $\delta$ -partition of $[ 0 , T ]$ , we have

$$
\begin{array} { r l } { \widehat { m } _ { T , 2 } ( x , y ) - \overline { { m } } _ { T , 2 } ( x , y ) = \displaystyle \frac { 1 } { n \mathbb { E } ( Z _ { 1 } ( x ) ) } \sum _ { j = 1 } ^ { n } \int _ { T _ { j - 1 } } ^ { T _ { j } } \left[ \zeta _ { t } \psi _ { y } ( Y _ { t } ) \Delta _ { t } ( x ) - \mathbb { E } \left\{ \zeta _ { t } \psi _ { y } ( Y _ { t } ) \Delta _ { t } ( x ) | \mathcal { F } _ { t - \delta } \right\} \right] \medskip \mathrm { d } t } & { { } } \\ { : = \displaystyle \frac { 1 } { n \mathbb { E } ( Z _ { 1 } ( x ) ) } \sum _ { j = 1 } ^ { n } \mathcal { L } _ { T , j } ( x , y ) . } & { { } } \end{array}
$$

Since $( \mathcal { L } _ { T , j } ( x , y ) ) _ { j \geq 1 }$ is a sequence of martingale differences with respect to the family $( \mathcal { F } _ { j - 1 } ) _ { j \geq 1 }$ then

$\mathbb { E } ( \mathcal { L } _ { T , j } ( x , y ) \times \mathcal { L } _ { T , k } ( x , y ) ) = 0$ for every $j , k \in \{ 1 , \dots n \}$ such that $j \neq k$ . Therefore (by ignoring the product term), we have

$$
I _ { T , 1 } = {  { \mathbb E } } \left[ \widehat m _ { T , 2 } ( x , y ) - \overline { m } _ { T , 2 } ( x , y ) \right] ^ { 2 } \simeq \frac { 1 } { n ^ { 2 } ( {  { \mathbb E } } ( Z _ { 1 } ( x ) ) ) ^ { 2 } } \sum _ { j = 1 } ^ { n } {  { \mathbb E } } ( \mathcal L _ { T , j } ( x , y ) ) ^ { 2 } .
$$

Using Jensen inequality and a double conditioning with respect to $\mathcal { S } _ { t - \delta , \delta }$ combined with (A3)(iii) –(iv), $I _ { T , 1 }$ may bounded as

$$
\begin{array} { l } { I _ { T , 1 } \leq \displaystyle \frac { 4 } { n ^ { 2 } ( \mathbb { E } ( Z _ { 1 } ( x ) ) ) ^ { 2 } } \sum _ { j = 1 } ^ { n } \int _ { T _ { j - 1 } } ^ { T _ { j } } \mathbb { E } \left[ \Delta _ { t } ^ { 2 } ( x ) p ( X _ { t } ) W _ { 2 } ( X _ { t } , y ) \right] \mathrm { d } t . } \\ { = \displaystyle \frac { 4 ( p ( x ) + o ( 1 ) ) ( W _ { 2 } ( x , y ) + o ( 1 ) ) [ M _ { 2 } f ( x ) + o ( 1 ) ] } { T \phi ( h _ { T } ) [ M _ { 1 } f ( x ) + o ( 1 ) ] ^ { 2 } } . } \end{array}
$$

Similarly, we have

$$
I _ { T , 2 } \leq \frac { 4 ( p ( x ) + o ( 1 ) ) [ M _ { 2 } f ( x ) + o ( 1 ) ] } { T \phi ( h _ { T } ) [ M _ { 1 } f ( x ) + o ( 1 ) ] ^ { 2 } } .
$$

Therefore

$$
\mathbb { E } ( Q _ { T } ^ { 2 } ( x , y ) ) \simeq I _ { T , 1 } + m ^ { 2 } ( x , y ) I _ { T , 2 } = \frac { 4 p ( x ) ( W _ { 2 } ( x , y ) + ( m ( x , y ) ) ^ { 2 } ) M _ { 2 } } { T \phi ( h _ { T } ) M _ { 1 } ^ { 2 } f ^ { 2 } ( x ) } .
$$

Moreover, using the decomposition (A2), Theorem 3.3 and Lemma 1.1 (in Chaouch and Laïb 2023) one can see that $\mathbb { E } ( Q _ { T } ^ { 2 } ( x , y ) )$ is negligible with respect to $\mathbb { E } ( Q _ { T } ^ { 2 } ( x , y ) )$ . This completes the proof.

Proof of Theorem 3.5: The proof of Theorem 3.5 is based essentially on Lemma A.4 established below, which gives the normality asymptotic of the principal term $Q _ { T } ( x , y )$ in (A3). Indeed, we have from (A3) that

$$
\overline { { { F \phi ( h _ { T } ) } } } \left( \widehat { m } _ { T } ( x , y ) - m ( x , y ) \right) = \sqrt { T \phi ( h _ { T } ) } B _ { T } ( x , y ) + \frac { \sqrt { T \phi ( h _ { T } ) } Q _ { T } ( x , y ) + \sqrt { T \phi ( h _ { T } ) } R _ { T } ( x , y ) } { \widehat { m } _ { T , 1 } ( x ) } .
$$

Under (A1)–(A3), Lemma 1.2 (in Chaouch and Laïb 2023) implies that $\widehat { m } _ { T , 1 } ( x )$ converges, almost surely, to $p ( x )$ as $T \to \infty$ . Moreover, using Lemma 1.3 (in Chaouch and Laïb 2023), we get under (A3)(i) $^ -$ (ii) combined with conditions (12) that $\sqrt { T \phi ( h _ { T } ) } B _ { T } ( x , y ) = \mathcal { O } _ { a . s . } ( h _ { T } ^ { \beta } \sqrt { T \phi ( h _ { T } ) } ) = o _ { a . s } ( 1 )$ , and

$$
\sqrt { T \phi ( h _ { T } ) } R _ { T } ( x , y ) = { \mathcal O } _ { a . s } \left( \sqrt { T \phi ( h _ { T } ) } h _ { T } ^ { \beta } \left( \frac { \log T } { T \phi ( h _ { T } ) } \right) ^ { 1 / 2 } \right) = { \mathcal O } _ { a . s . } \left( h _ { T } ^ { \beta } \log T ^ { 1 / 2 } \right) = o _ { a . s } ( 1 ) .
$$

The proof may be then achieved by Lemma A.4 and Slutsky’s Theorem.

Lemma A.4: Under conditions (A1)–(A3), we have

$$
\begin{array} { r l } & { \sqrt { T \phi ( h _ { T } ) } ( \widehat { m } _ { T } ( x , y ) - m ( x , y ) ) \xrightarrow { d } N ( 0 , \tilde { \sigma } ^ { 2 } ( x , y ) ) \quad \mathrm { a s ~ } T \longrightarrow \infty } \\ & { \mathrm { w h e r e } \quad \tilde { \sigma } ^ { 2 } ( x , y ) \le \displaystyle \frac { M _ { 2 } } { M _ { 1 } ^ { 2 } f ( x ) \hat { P } ( x ) } \overline { { W } } _ { 2 } ( x , y ) . } \end{array}
$$

Proof of Lemma A.4: We have

$$
\begin{array} { l } { \sqrt { T \phi ( h _ { T } ) } Q _ { T } ( x , y ) = \displaystyle \sum _ { i = 1 } ^ { n } \xi _ { T , i } ( x , y ) , \quad \mathrm { w i t h } \xi _ { T , i } ( x , y ) = \eta _ { T , i } ( x , y ) - { \mathbb E } \left[ \eta _ { T , i } ( x , y ) \left| { \mathcal F } _ { t - \delta } \right] \right. } \\ { \displaystyle \left. \eta _ { T , i } ( x , y ) = \frac { 1 } { { \mathbb E } Z _ { 1 } } \sqrt { \frac { \phi _ { T } ( h ) } { n } } \int _ { T _ { i - 1 } } ^ { T _ { i } } \zeta _ { t } \Delta _ { t } ( x ) \left[ \psi _ { y } ( Y _ { t } ) - m ( x , y ) \right] \mathrm { d } t . \right. } \end{array}
$$

Since for any $i \geq 1$ and $t \in [ T _ { i - 1 } , T _ { i } ] , \mathcal { F } _ { i - 2 } \subset \mathcal { F } _ { t - \delta } \subset \mathcal { F } _ { i - 1 }$ , then $( \xi _ { T , i } ( x , y ) ) _ { i \geq 1 }$ is $\mathcal { F } _ { i - 1 }$ -measurable, $\mathbb { E } ( | \xi _ { T , i } | ) < \infty$ provided $\mathbb { E } ( \zeta _ { t } ^ { 2 } ) < \infty$ and $\mathbb { E } ( X _ { t } ^ { 2 } ) < \infty$ . Moreover, we have for any $1 \leq i \leq n$ , $\mathbb { E } ( \xi _ { T , i } \ | \mathcal { F } _ { i - 2 } ) = \mathbb { E } \{ \mathbb { E } [ \eta _ { i } \ | \mathcal { F } _ { t - \delta } ] | \mathcal { F } _ { i - 2 } \} - \mathbb { E } \{ \mathbb { E } [ \eta _ { i } \ | \mathcal { F } _ { t - \delta } ] | \mathcal { F } _ { i - 2 } \} = 0$ a.s. Hence $( \xi _ { T , i } ( x , y ) ) _ { i \geq 1 }$ is a sequence of martingale differences with respect to the $\sigma$ -fields $( \mathcal { F } _ { i - 1 } ) _ { i \geq 1 }$ . To prove the asymptotic normality, it suffices to check both following conditions (see Corollary 3.1, p. 56, Hall and Heyde 1980):

(a) $\begin{array} { r } { ) \sum _ { i = 1 } ^ { n } \mathbb { E } [ \xi _ { T , i } ^ { 2 } ( x , y ) | \mathcal { F } _ { i - 2 } ] \overset { \mathbb { P } } {  } \tilde { \sigma } ^ { 2 } ( x , y ) \mathrm { a n d } ( \mathrm { b } ) n \mathbb { E } [ \xi _ { T , i } ^ { 2 } ( x , y ) \mathbf { 1 } _ { \{ | \xi _ { T , i } ( x , y ) | > \epsilon \} } ] = o ( 1 ) | \mathcal { F } _ { i - 1 } [ \xi _ { T , i } ( x , y ) ] | \mathcal { F } _ { i - 1 } [ \xi _ { T , i } ( x , y ) ] , } \end{array}$ holds for any $\epsilon > 0$ .

Proof of (a) Observe now that

$$
| \sum _ { i = 1 } ^ { n } \mathbb { E } [ \eta _ { T , i } ^ { 2 } ( x , y ) | \mathcal { F } _ { i - 2 }  ] - \sum _ { i = 1 } ^ { n } \mathbb { E } [ \xi _ { T , i } ^ { 2 } ( x , y ) | \mathcal { F } _ { i - 2 }  ] | \leq \Big | \sum _ { i = 1 } ^ { n } ( \mathbb { E } [ \eta _ { T , i } ( x , y ) | \mathcal { F } _ { i - 2 }  ] ) ^ { 2 } | .
$$

Using (A1), (A3)(i), $\mathrm { \cdot } ( \mathrm { i } ^ { \prime } )$ , (ii) and (iv) with Lemma A.1, and a double conditioning with respect to the $\sigma$ -field $S _ { t - \delta , \delta }$ and the fact that $n \mathbb { E } Z _ { 1 } ( t ) = O ( T \phi ( h ) )$ , we have

$$
\begin{array} { r l } & { \left| \mathbb { E } \left( \eta _ { T , i } \mid \mathcal { F } _ { i - 2 } \right) \right| } \\ & { \quad = \frac { 1 } { \mathbb { E } Z _ { 1 } } \sqrt { \frac { \phi _ { T } ( h ) } { n } } \left| \int _ { T _ { i - 1 } } ^ { T _ { i } } \mathbb { E } ( p ( X _ { t } ) \Delta _ { t } ( x ) \left[ m ( X _ { t } , y ) - m ( x , y ) \right] \mid \mathcal { F } _ { i - 2 } ) \mathrm { d } t \right| } \\ & { \quad \le \sqrt { n \phi _ { T } ( h ) } \frac { p ( x ) } { n \mathbb { E } Z _ { 1 } } \underset { u \in B ( x , h ) } { \operatorname* { s u p } } \left| m ( u , y ) - m ( x , y ) \right| \underset { u \in B ( x , h ) } { \operatorname* { s u p } } \left| p ( u ) - p ( x ) \right| \left| \int _ { T _ { i - 1 } } ^ { T _ { i } } \mathbb { E } ( \Delta _ { t } ( x ) \mid \mathcal { F } _ { i - 2 } ) \mathrm { d } t \right| } \end{array}
$$

$$
\begin{array} { l } { \displaystyle = O \left( \sqrt { n \phi _ { T } ( h ) } h ^ { \beta } \right) O \left( \phi ( h _ { T } ) \int _ { T _ { i - 1 } } ^ { T _ { i } } f _ { t , T _ { i - 2 } } ( x ) \mathrm { d } t + o ( 1 ) \right) \frac { 1 } { n \mathbb { E } Z _ { 1 } } } \\ { \displaystyle = O \left( \sqrt { n \phi _ { T } ( h ) } h ^ { \beta } \right) O \left( \frac { 1 } { T } \int _ { T _ { i - 1 } } ^ { T _ { i } } b _ { t , \alpha _ { 0 } } ( x ) \mathrm { d } t \right) . } \end{array}
$$

It follows by (A2)-(iii) and the Cauchy–Schwarz inequality that

$$
\sum _ { i = 1 } ^ { n } { \big ( } \mathbb { E } \left[ \eta _ { T , i } ( x , y ) \mid { \mathcal { F } } _ { i - 2 } \right] { \big ) } ^ { 2 } = O ( h ^ { 2 \beta } \phi ( h ) ) = o ( 1 ) .
$$

Thus we have only to show that $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \mathbb { E } [ \eta _ { T , i } ^ { 2 } ( x , y ) \vert \mathcal { F } _ { i - 2 } ] \stackrel { \mathbb { P } } {  } \sigma ^ { 2 } ( x , y ) } \end{array}$ . Using again the Cauchy– Schwarz inequality, one may write

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { n } \mathbb { E } [ \eta _ { T _ { i } } ^ { 2 } ( x , y ) | \mathcal { F } _ { i - 2 } ] = \frac { 1 } { ( \mathbb { E } Z _ { 1 } ) ^ { 2 } } \frac { \phi _ { T } ( h ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } [ ( \int _ { T _ { i - 1 } } ^ { T _ { i } } \zeta \mathbf { A } _ { t } ( x ) [ \psi _ { y } ( Y _ { t } ) - m ( x , y ) ] ) ^ { 2 } \Bigg | \mathcal { F } _ { i - 2 } ] } } \\ & { } & { \leq \frac { \delta } { ( \mathbb { E } Z _ { 1 } ) ^ { 2 } } \frac { \phi _ { T } ( h ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } [ \int _ { T _ { i - 1 } } ^ { T _ { i } } \zeta _ { t } ^ { 2 } \Delta _ { t } ^ { 2 } ( x ) [ \psi _ { y } ( Y _ { t } ) - m ( x , y ) ] ^ { 2 } \mathrm { d } t \Bigg | \mathcal { F } _ { i - 2 } ] } \\ & { } & { = \frac { \delta } { ( \mathbb { E } Z _ { 1 } ) ^ { 2 } } \frac { \phi _ { T } ( h ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } [ \int _ { T _ { i - 1 } } ^ { T _ { i } } \zeta _ { t } ^ { 2 } \Delta _ { t } ^ { 2 } ( x ) [ \psi _ { y } ( Y _ { t } ) - m ( X _ { t } , y ) ] ^ { 2 } \mathrm { d } t \Bigg | \mathcal { F } _ { i - 2 } ] } \\ & { } & { + \frac { \delta } { ( \mathbb { E } Z _ { 1 } ) ^ { 2 } } \frac { \phi _ { T } ( h ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } [ \int _ { T _ { i - 1 } } ^ { T _ { i } } \zeta _ { t } ^ { 2 } \Delta _ { t } ^ { 2 } ( x ) [ m ( X _ { t } , y ) - m ( x , y ) ] ^ { 2 } \mathrm { d } t \Bigg | \mathcal { F } _ { i - 1 } ] } \\ & { } & { = : A _ { n } + C _ { n } . } \end{array}
$$

Now, let us evaluate the term $A _ { n }$ . Conditioning three times with respect to $\mathcal { F } _ { t - \delta }$ and $S _ { t - \delta } ,$ and making use of Conditions (A3) $\mathrm { ( i ^ { \prime } ) }$ , (iii), (iv) and the fact that $T = n \delta$ , to get from Lemma A.1 that

$$
\begin{array} { r l } { A _ { n } = \displaystyle \frac { \delta } { ( \mathbb { E } \mathbb { Z } _ { 1 } ) ^ { 2 } } \displaystyle \frac { \phi _ { T } ( h ) } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } \bigg [ \int _ { r _ { i - 1 } } ^ { \pi _ { i } } P ( X _ { i } ) \Delta _ { t } ^ { 2 } ( x ) \overline { { W _ { 2 } } } ( X _ { t } , y ) \mathrm { d } t \bigg | \mathcal { F } _ { i - 2 } \bigg ] } & { } \\ { \leq \displaystyle \frac { \delta } { ( \mathbb { E } \mathbb { Z } _ { 1 } ) ^ { 2 } } \displaystyle \frac { \phi _ { T } ( h ) } { n } ( \rho ( x ) + \sigma ( 1 ) ) ( \overline { { W _ { 2 } } } ( x , y ) + \sigma ( 1 ) ) \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } \bigg [ \int _ { r _ { i - 1 } } ^ { \pi _ { i } } \Delta _ { t } ^ { 2 } ( x ) \mathrm { d } t \bigg | \mathcal { F } _ { i - 2 } \bigg ] } & { } \\ { \leq \displaystyle ( \delta + o ( 1 ) ) \rho ( x ) \overline { { W } } _ { 2 } ( x , y ) \displaystyle \frac { \phi _ { T } ^ { 2 } ( h ) } { ( \mathbb { E } \mathbb { Z } _ { 1 } ) ^ { 2 } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \int _ { r _ { i - 1 } } ^ { \pi _ { i } } \mathbb { E } \bigg [ \frac { 1 } { \phi _ { T } ( h ) } \Delta _ { t } ^ { 2 } ( x ) \bigg | \mathcal { F } _ { i - 2 } \bigg ] \mathrm { d } t } & { } \\ { \leq \delta ( \delta + o ( 1 ) ) \rho ( x ) \overline { { W } } _ { 2 } ( x ) \displaystyle \frac { \phi _ { T } ^ { 2 } ( h ) } { ( \mathbb { E } \mathbb { Z } _ { 1 } ) ^ { 2 } } M _ { 2 } \bigg \{ \frac { 1 } { T } \sum _ { i = 1 } ^ { n } \int _ { r _ { i - 1 } } ^ { \pi _ { i } } \int _ { t , \tilde { T } _ { i - 2 } ( x ) \mathrm { d } t } ^ { \pi _ { i } } } & { } \\  + o _ { \lambda } \displaystyle [ \frac { 1 } { T } \sum _ { i = 1 } ^ { n } \int _ { r _ { i - 1 } } ^ { \pi _ { i } } \frac { \delta _ { T } ( x _ { i - 2 } , x _ { i } ( h ) ) }  \end{array}
$$

The Riemann’s sum combined with condition (A2)(iii) gives that

$$
{ \frac { 1 } { T } } \sum _ { i = 1 } ^ { n } \int _ { T _ { i - 1 } } ^ { T _ { i } } f _ { t , T _ { i - 2 } } ( x ) \mathrm { d } t \leq { \frac { 1 } { T } } \int _ { 0 } ^ { T } f _ { t , T _ { t - { \delta } } } ( x ) \mathrm { d } t \longrightarrow f ( x ) \quad { \mathrm { a . s . ~ a s ~ } } T \longrightarrow \infty .
$$

Moreover, by (A2)(ii) one gets $\begin{array} { r } { \frac { g _ { t , T _ { i - 2 } , x } ( h _ { T } ) } { \phi _ { T } ( h ) } = o ( 1 ) } \end{array}$ as $T \longrightarrow \infty$ . Therefore, we have

$$
\begin{array} { l } { { A _ { n } \leq \delta ( \delta + o ( 1 ) ) p ( x ) \overline { { { W } } } _ { 2 } ( x , y ) \frac { \phi _ { T } ^ { 2 } ( h ) } { ( \delta \phi _ { T } ( h _ { T } ) M _ { 1 } f ( x ) + o ( 1 ) ) ^ { 2 } } M _ { 2 } \left[ f ( x ) + o ( 1 ) \right] } } \\ { { \mathrm { } = \frac { M _ { 2 } } { M _ { 1 } ^ { 2 } f ( x ) } \rlap / P ( x ) \overline { { { W } } } _ { 2 } ( x , y ) : = \widetilde { \sigma } ^ { 2 } ( x , y ) \mathrm { a s } T \longrightarrow \infty . } } \end{array}
$$

On the other hand, by the same arguments as above combined with the fact that $\operatorname* { s u p } _ { u \in B ( x , h ) } | m ( x ) -$ $m ( u ) | = O ( h ^ { 2 \beta } ) _ { a . s . } ,$ we get $C _ { n } = o _ { a . s . } ( 1 )$ .

Proof of part $( b )$ Using successively Hölder, Markov, Jensen and Minkowski inequalities combined with conditions (A3)(iii), (A3)(iv) and Lemma A.1, we get, for any $\epsilon > 0$ and fixed real numbers $/ > 1$ and $q > 1$ such that $1 / p + 1 / q = 1$ ,

$n \mathbb { E } [ \xi _ { T , i } ^ { 2 } ( x , y ) 1 _ { \{ | \xi _ { T , i } ( x , y ) | > \epsilon \} } ] \le 4 n ( \epsilon / 2 ) ^ { - 2 q / P } \mathbb { E } [ | \eta _ { T , i } | ^ { 2 q } ] = O \left( ( T \phi ( h _ { T } ) ) ^ { - \gamma / 2 } \right) = o _ { a , s } ( 1 )$ by taking $2 q = 2 + \gamma$ $0 < \gamma \ < 1 ,$ ), since $T \phi ( h _ { T } )$ towards to infinity as $T$ goes to infinity.

Proof of Corollary 3.7: Observe that

$$
\begin{array} { r l } & { \sqrt { \frac { T F _ { x , T } ( h _ { T } ) } { \widetilde { V } _ { T } ^ { 2 } ( x , y ) } } \left( \widehat { m } _ { T } ( x , y ) - m ( x , y ) \right) } \\ & { \quad = \sqrt { \frac { F _ { x , T } ( h _ { T } ) } { \phi ( h _ { T } ) f ( x ) } } \sqrt { \frac { \sigma ^ { 2 } ( x , y ) f ( x ) } { \widetilde { V } _ { T } ^ { 2 } ( x , y ) } } \sqrt { \frac { T \phi ( h _ { T } ) } { \sigma ^ { 2 } ( x , y ) } } \left( \widehat { m } _ { T } ( x , y ) - m ( x , y ) \right) . } \end{array}
$$

It follows from the consistency of $F _ { x , T } ( h _ { T } )$ and (A2)(i) that $\frac { F _ { x , T } ( h _ { T } ) } { \phi ( h _ { T } ) f ( x ) }$ goes to 1 a.s. as $T$ goes to infinity. By Theorem 3.5, the quantity $\sqrt { \frac { T \phi ( h _ { T } ) } { \sigma ( x , y ) } } ( \widehat { m } _ { T } ( x , y ) - m ( x , y ) )$ converges to $N ( 0 , 1 )$ as $T \to \infty$ . Then using the non-decreasing property of the cumulative standard normal distribution function $\Psi$ , we get, for a given risk $0 < \alpha < 1$ , the $( 1 - \alpha )$ - pseudo-confidence interval

$$
\sqrt { \frac { T \phi ( h _ { T } ) } { \sigma ^ { 2 } ( x , y ) } } \bigg | \widehat { m } _ { T } ( x , y ) - m ( x , y ) \bigg | \leq \Psi ^ { - 1 } \left( 1 - \frac { \alpha } { 2 } \right) .
$$

Considering now the statement (13) combined with Proposition 3.6, it holds that

$$
\operatorname* { l i m } _ { T \to \infty } \frac { \sigma ^ { 2 } ( x , y ) f ( x ) } { \widetilde { V } _ { n } ^ { 2 } ( x , y ) } \leq \operatorname* { l i m } _ { T \to \infty } \frac { f ( x ) V ^ { 2 } ( x , y ) } { \widetilde { V } _ { n } ^ { 2 } ( x , y ) } = \operatorname* { l i m } _ { T \to \infty } \widetilde { \overline { { V } } } _ { n } ^ { 2 } ( x , y ) = 1 \quad \mathrm { a . s . } ,
$$

since $\widetilde { V } _ { n } ^ { 2 } ( x , y )$ is a consistent estimator of $\widetilde { V } ^ { 2 } ( x , y )$ . The proofs follows then from the statements (A13), (A14) and (A15).