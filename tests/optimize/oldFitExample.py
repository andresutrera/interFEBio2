import interFEBio
import numpy as np
from scipy.interpolate import interp1d
import warnings
import xml.etree.ElementTree as ET
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OMP_NUM_THREADS"] = "8"
####### FUNCTION TO DETERMINE THE SIMULATION RESULTS #######
####### written in terms of the xplt class
####### self,file arguments are needed.
####### Must return a x,y lists, equivalent to the experimental measurements (xexp,yexp)
####### In this case, the same function works for both simulations, but it could be one function per simulation
def ringResult(self,file):
    xplt = interFEBio.xplt(file)

    xplt.readAllStates()
    #xplt.clearDict()
    #xplt.reader.file.close()
    #print(xplt.dictionary)
    surfaceID = xplt.mesh.surfaceID('contactPin')
    surfNode = xplt.mesh.surface[surfaceID].faces[1][0]
    disp = xplt.results['displacement'].getData(domain=0)[:,surfNode-1,1]*2
    dispTime = xplt.results['displacement'].getDataTime()
    contactForce = xplt.results['contact force'].getData(domain=surfaceID)[:,0,1]*-4.0
    contactTime = xplt.results['contact force'].getDataTime()

    time = np.linspace(1.0,2.0,100)

    dispInterp = interp1d(dispTime,disp,fill_value='extrapolate')(time)
    forceInterp = interp1d(contactTime,contactForce,fill_value='extrapolate')(time)

    return dispInterp,forceInterp
####### FUNCTION TO DETERMINE THE SIMULATION RESULTS #######

    

def longResult(self,file):
    xplt = interFEBio.xplt(file)

    xplt.readAllStates()
    #xplt.clearDict()

    lamb = xplt.results['displacement'].getData(domain=0)[:,1,0]+1
    stress = xplt.results['stress'].getData(domain=0)[:,0,0]
    time = xplt.time

    timeInterp = np.linspace(0.0,1.0,100)
    lambInterp = interp1d(time,lamb,fill_value='extrapolate')(timeInterp)
    stressInterp = interp1d(time,stress,fill_value='extrapolate')(timeInterp)
    return lambInterp,stressInterp
####### FUNCTION TO DETERMINE THE SIMULATION RESULTS #######

# res = longResult("a", 'long.xplt')
# print(res)

fit = interFEBio.fit(skip=0)


circExp = np.loadtxt('AortaAbdominalAnilloData_force.txt')
circExp[:,1] = abs(circExp[:,1])
longExp = np.loadtxt('AortaAbdominalLongData_stress.txt')
longExp[:,1] = abs(longExp[:,1])


dispVect = np.linspace(circExp[0,0],circExp[-1,0],num=200)
lambdVect = np.linspace(longExp[0,0],longExp[-1,0],num=200)

circFx = interp1d(circExp[:,0],circExp[:,1])
longFx = interp1d(longExp[:,0],longExp[:,1])

circDataFinal = np.column_stack((dispVect,circFx(dispVect)))
longDataFinal = np.column_stack((lambdVect,longFx(lambdVect)))

w = interp1d([0,100],[1,1])

#print(longExp)
fit.addCase('ring',1,'ring.feb','ring',circDataFinal,ringResult,w)
fit.addCase('long',1,'long.feb','long',longDataFinal,longResult,w)

# fit.addTask('close',1,'close.feb','close')
# fit.addTaskFcn(name='close', fcn=taskFcn)


fit.p.add('nu_c', 0)
fit.p.add('nu_k1', 0)
fit.p.add('nu_k2', 0)
fit.p.add('nu_kappa',0)
fit.p.add('nu_gamma', 0)

fit.p.add('c', expr='0.01923184*2**nu_c',min=0, max=50E-3)
fit.p.add('k1', expr='0.18402581*2**nu_k1',min=0, max=500E-3)
fit.p.add('k2', expr='1.52815874*2**nu_k2',min=0, max=4.0)
fit.p.add('kappa',expr='0.20320722*2**nu_kappa',min=0.05, max=0.28)
fit.p.add('gamma', expr='51.3898438*2**nu_gamma',min=20, max=75)
fit.p.add('k', expr='100*k1')

fit.optimize(method='bfgs',options={'eps': 0.02})




########### LM METHOD ##############
# fit.p.add('c', 0.00683918, min=0, max=50E-3)
# fit.p.add('k1', 0.07370464, min=0, max=500E-3)
# fit.p.add('k2', 0.49885124, min=0, max=4.0)
# fit.p.add('kappa',0.11643317, min=0.05, max=0.28)
# fit.p.add('gamma', 56.3248843, min=20, max=75)
# fit.p.add('k', expr='1000*c')
# fit.optimize(xtol=1.e-20,ftol=1.e-20,epsfcn=0.0035)


#fit.optimize(method='basinhopping')
