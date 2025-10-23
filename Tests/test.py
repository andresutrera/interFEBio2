from interFEBio.XPLT.XPLT import xplt

xp = xplt("ring.xplt")

xp.readSteps([0, 200])
print(xp.dictionary)
str = xp.results["displacement"]

print(str.time(slice(0, None)).nodes(10).comp("x"))
print(str[:, 10, 0])

ff = str = xp.results["contact force"]

reee = ff.surface("contactPin").time(":").faces(0).comp("x")

print(reee.shape)

print(xp.results.times())
