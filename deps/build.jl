using ADCME

rm("$(@__DIR__)/CustomOp/build", force=true, recursive=true)
mkdir("$(@__DIR__)/CustomOp/build")
cd("$(@__DIR__)/CustomOp/build")
ADCME.cmake()
ADCME.make()