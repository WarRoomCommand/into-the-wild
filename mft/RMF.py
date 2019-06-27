class RegularizedMeanField(Solver):
    """Implementation of regularized mean field method for solving the inverse Ising
    problem, as described in Daniels, Bryan C., David C. Krakauer, and Jessica C. Flack.
    ``Control of Finite Critical Behaviour in a Small-Scale Social System.'' Nature
    Communications 8 (2017): 14301.  doi:10.1038/ncomms14301
    
    Specific to pairwise Ising constraints.
    """
    def __init__(self, *args, **kwargs):
        """
        See Solver. Default sample_method is '_metropolis'.
        """
        super(RegularizedMeanField,self).__init__(*args,**kwargs)
        # some case handling to ensure that RMF gets control over the random number generator
        self.setup_sampler(kwargs.get('sample_method','metropolis'))

    def solve(self, samples,
              sample_size=100000,
              seed=0,
              change_seed=False,
              min_size=0,
              min_covariance=False,
              min_independent=True,
              cooc_cov=None,
              priorLmbda=0.,
              bracket=None,
              n_grid_points=200):
        """Varies the strength of regularization on the mean field J to best fit given
        cooccurrence data.
        
        n_grid_points : int, 200
            If bracket is given, first test at n_grid_points points evenly spaced in the
            bracket interval, then give the lowest three points to
            scipy.optimize.minimize_scalar
        sample_size : int, 100_000
        seed : int, 0
            initial seed for rng, seed is incremented by mean_field_ising.seedGenerator if
            change Seed option is True
        change_seed : bool, False
        min_size : int, 0
            Use a modified model in which samples with fewer ones than min_size are not
            allowed.
        min_covariance : bool, False
            ** As of v1.0.3, not currently supported **
            Minimize covariance from emperical frequencies (see notes); trying to avoid
            biases, as inspired by footnote 12 in TkaSchBer06
        min_independent : bool, True
            ** As of v1.0.3, min_independent is the only mode currently supported **
            Each <xi> and <xi xj> residual is treated as independent
        cooc_cov : ndarray,None
            ** As of v1.0.3, not currently supported **
            Provide a covariance matrix for residuals.  Should typically be
            coocSampleCovariance(samples).  Only used if min_covariance and
            min_independent are False.
        priorLmbda : float,0.
            ** As of v1.0.3, not currently implemented **
            Strength of noninteracting prior.
        """

        from scipy import transpose

        numDataSamples = len(samples)
        # convert input to coocMat
        coocMatData = mean_field_ising.cooccurrence_matrix((samples+1)/2)
        
        if cooc_cov is None:
            cooc_cov = mean_field_ising.coocSampleCovariance(samples)
        
        if change_seed: seedIter = mean_field_ising.seedGenerator(seed, 1)
        else: seedIter = mean_field_ising.seedGenerator(seed, 0)
        
        if priorLmbda != 0.:
            raise NotImplementedError("priorLmbda is not currently supported")
            lmbda = priorLmbda / numDataSamples

        # stuff defining the error model, taken from findJmatrixBruteForce_CoocMat
        # 3.1.2012 I'm pretty sure the "repeated" line below should have the transpose, but
        # coocJacobianDiagonal is not sensitive to this.  If you use non-diagonal jacobians in the
        # future and get bad behavior you may want to double-check this.
        if min_independent:
            coocStdevs = mean_field_ising.coocStdevsFlat(coocMatData,numDataSamples)
            coocStdevsRepeated = ( coocStdevs*np.ones((len(coocStdevs),len(coocStdevs))) ).T
        elif min_covariance:
            raise Exception("min_covariance is not currently supported")
            empiricalFreqs = np.diag(coocMatData)
            covTildeMean = covarianceTildeMatBayesianMean(coocMatData,numDataSamples)
            covTildeStdevs = covarianceTildeStdevsFlat(coocMatData,numDataSamples,
                empiricalFreqs)
            covTildeStdevsRepeated = (
                    covTildeStdevs*np.ones((len(covTildeStdevs),len(covTildeStdevs))) ).T
        else:
            raise NotImplementedError("correlated residuals calculation is not currently supported")
            if cooc_cov is None: raise Exception
            cov = cooc_cov # / numDataSamples (can't do this here due to numerical issues)
                          # instead include numDataSamples in the calculation of coocMatMeanZSq

        # for use in gammaPrime <-> priorLmbda
        freqsList = np.diag(coocMatData)
        pmean = np.mean(freqsList)
        
        # adapted from findJMatrixBruteForce_CoocMat
        def samples(J):
           seed = next(seedIter)
           if min_covariance:
               J = tildeJ2normalJ(J, empiricalFreqs)
           burninDefault = 100*self.n
           J = J + J.T
           self.multipliers = np.concatenate([J.diagonal(), squareform(mean_field_ising.zeroDiag(-J))])
           self.sampler.rng = np.random.RandomState(seed)
           self.generate_samples(1, burninDefault, sample_size=int(sample_size))
           isingSamples = self.samples.copy()
           return isingSamples

        # adapted from findJMatrixBruteForce_CoocMat
        def func(meanFieldGammaPrime):
            
            # translate gammaPrime prior strength to lambda prior strength
            meanFieldPriorLmbda = meanFieldGammaPrime / (pmean**2 * (1.-pmean)**2)
            
            # calculate regularized mean field J
            J = mean_field_ising.JmeanField(coocMatData,
                                            meanFieldPriorLmbda=meanFieldPriorLmbda,
                                            numSamples=numDataSamples)

            # sample from J
            isingSamples = samples(J)
            
            # calculate residuals, including prior if necessary
            if min_independent: # Default
                dc = mean_field_ising.isingDeltaCooc(isingSamples, coocMatData)/coocStdevs
            elif min_covariance:
                dc = isingDeltaCovTilde(isingSamples, covTildeMean, empiricalFreqs)/covTildeStdevs
            else:
                dc = mean_field_ising.isingDeltaCooc(isingSamples, coocMatMean)
                if priorLmbda != 0.:
                    freqs = np.diag(coocMatData)
                    factor = np.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                    factorFlat = aboveDiagFlat(factor)
                    priorTerm = lmbda * factorFlat * flatJ[ell:]**2
                
                dc = np.concatenate([dc,priorTerm])
                
            if self.verbose:
                print("RegularizedMeanField.solve: Tried "+str(meanFieldGammaPrime))
                print("RegularizedMeanField.solve: sum(dc**2) = "+str(np.sum(dc**2)))
                
            return np.sum(dc**2)

        if bracket is not None:
            gridPoints = np.linspace(bracket[0], bracket[1], n_grid_points)
            gridResults = [ func(p) for p in gridPoints ]
            gridBracket = self.bracket1d(gridPoints, gridResults)
            solution = minimize_scalar(func, bracket=gridBracket)
        else:
            solution = minimize_scalar(func)

        gammaPrimeMin = solution['x']
        meanFieldPriorLmbdaMin = gammaPrimeMin / (pmean**2 * (1.-pmean)**2)
        J = mean_field_ising.JmeanField(coocMatData,
                                        meanFieldPriorLmbda=meanFieldPriorLmbdaMin,
                                        numSamples=numDataSamples)
        J = J + J.T

        # convert J to {-1,1} basis
        h = -J.diagonal()
        J = -mean_field_ising.zeroDiag(J)
        self.multipliers = convert_params( h, squareform(J)*2, '11', concat=True )

        return self.multipliers

    def bracket1d(self, xList, funcList):
        """Assumes xList is monotonically increasing
        
        Get bracketed interval (a,b,c) with a < b < c, and f(b) < f(a) and f(c).
        (Choose b and c to make f(b) and f(c) as small as possible.)
        
        If minimum is at one end, raise error.
        """

        gridMinIndex = np.argmin(funcList)
        gridMin = xList[gridMinIndex]
        if (gridMinIndex == 0) or (gridMinIndex == len(xList)-1):
            raise Exception("Minimum at boundary")
        gridBracket1 = xList[ np.argmin(funcList[:gridMinIndex]) ]
        gridBracket2 = xList[ gridMinIndex + 1 + np.argmin(funcList[gridMinIndex+1:]) ]
        gridBracket = (gridBracket1,gridMin,gridBracket2)
        return gridBracket
#end RegularizedMeanField