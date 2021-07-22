@echo off

set workPath=%~dp0
set thirdPartyPath=%~dp03rd

if exist %thirdPartyPath% (
    echo %thirdPartyPath% exist
    rd /S /Q %thirdPartyPath%
)

cd %workPath%
if exist %workPath%build (
  rd /S /Q %workPath%build
)
if exist %workPath%\output (
  rd /S /Q %workPath%\output
)
