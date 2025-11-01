import matplotlib.pyplot as plt
from interFEBio.XPLT import xplt

file = xplt("../../PreProcessing/Contact/simpleContact.xplt")
file.readAllStates()

# Results dictionary
print(file.dictionary)

# Results summary
print(file.results)

# Contact force
force = file.results["contact force"].region("bottom_+z").time(":").comp("z")
time = file.results.times()

plt.plot(time, force)
plt.show()
