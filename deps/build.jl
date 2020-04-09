using ADCME

cd("$(@__DIR__)/CustomOp")
rm("build", force=true, recursive=true)
mkdir("build")
cd("build")
ADCME.cmake()
ADCME.make()