@echo off
set PADDLE_DIR=/path/to/Paddle/include

set workPath=%~dp0
set thirdPartyPath=%~dp03rd

if exist %thirdPartyPath% (
    echo %thirdPartyPath% exist
    rd /S /Q %thirdPartyPath%
)

echo createDir %thirdPartyPath%
md %thirdPartyPath%  


cd %thirdPartyPath%
wget --no-check-certificate https://bj.bcebos.com/paddlex/tools/openssl-1.1.0k.tar.gz
tar -zxvf openssl-1.1.0k.tar.gz
del openssl-1.1.0k.tar.gz

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

cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DPADDLE_DIR=%PADDLE_DIR%

cd %workPath%
