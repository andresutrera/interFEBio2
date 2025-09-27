from interFEBio import Core, Materials

F0 = Core.parameters.FEParamMat3d(
    Core.mat3d(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
)


ps = Materials.FEPreStrainUncoupledElastic(
    prestrain=Materials.FEConstPrestrainGradient(f0=F0, ramp=1.0)
)

nh = Materials.FEIncompNeoHookean(g=1.0)
mat = Materials.FEPreStrainUncoupledElastic(elastic=nh, prestrain=ps)


print(mat.to_xml_string())