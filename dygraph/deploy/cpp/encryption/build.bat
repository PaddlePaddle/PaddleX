@echo off

set workPath=%~dp0
set thirdPartyPath=%~dp03rd

if exist %thirdPartyPath% (
    echo %thirdPartyPath% exist
    rd /S /Q %thirdPartyPath%
)

echo createDir %thirdPartyPath%
md %thirdPartyPath%  


cd %thirdPartyPath%
wget --no-check-certificate https://bj.bcebos.com/paddlex/tools/windows_openssl-1.1.0k.zip
tar -zxvf windows_openssl-1.1.0k.zip
del windows_openssl-1.1.0k.zip



cd %workPath%
if exist %workPath%build (
  rd /S /Q %workPath%build
)
if exist %workPath%\output (
  rd /S /Q %workPath%\output
)

MD %workPath%build
MD %workPath%\output
cd %workPath%build

cmake .. -G "Visual Studio 16 2019" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release

cd %workPath%
