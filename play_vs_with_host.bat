@echo off
chcp 65001 >nul

REM =============================================================================
REM Go Game Battle Script with Custom Host
REM Usage: play_vs_with_host.bat [rounds] [my_player] [opponent_player] [host_path]
REM   rounds: Total rounds to play (default: 4, must be even)
REM   my_player: Your player file name without extension (default: my_player)
REM   opponent_player: Opponent player file name without extension (default: opponent/random_player_0309)
REM   host_path: Path to the host/referee program (default: host)
REM Example:
REM   play_vs_with_host.bat 4 my_player random_player host
REM   play_vs_with_host.bat 4 my_player random_player host_gui
REM =============================================================================

REM Step 1: Parse command line arguments
set "play_time=4"
set "my_player=opponent\my_player_xjy_final"
set "opponent_player=my_player"
set "host_path=host"

if not "%1"=="" set "play_time=%1"
if not "%2"=="" set "my_player=%2"
if not "%3"=="" set "opponent_player=%3"
if not "%4"=="" set "host_path=%4"

REM Validate play_time is even
set /a check_even=%play_time% %% 2
if not "%check_even%"=="0" (
    echo Error: Number of rounds must be even
    echo Example: play_vs_with_host.bat 4 my_player random_player host
    exit /b 1
)

REM Step 2: Clean up previous files
del /f /q *.so 2>nul
del /f /q *.class 2>nul
del /f /q exe.exe 2>nul
del /f /q exe_opponent.exe 2>nul

REM Step 3: Detect host programming language
echo Detecting host: %host_path%...

set "host_cmd="
set "host_ext="
set "host_python=false"

if exist "%host_path%.py" (
    set "host_cmd=python %host_path%.py"
    set "host_ext=.py"
    set "host_python=true"
    echo   Found: %host_path%.py (Python)
    goto :detect_my_player
)

if exist "%host_path%3.py" (
    set "host_cmd=python %host_path%3.py"
    set "host_ext=3.py"
    set "host_python=true"
    echo   Found: %host_path%3.py (Python3)
    goto :detect_my_player
)

if exist "%host_path%.cpp" (
    g++ -O2 %host_path%.cpp -o host_temp.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++ compilation failed for %host_path%.cpp
        exit /b 1
    )
    set "host_cmd=host_temp.exe"
    set "host_ext=.cpp"
    echo   Found: %host_path%.cpp (C++)
    goto :detect_my_player
)

if exist "%host_path%11.cpp" (
    g++ -std=c++11 -O2 %host_path%11.cpp -o host_temp.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++11 compilation failed for %host_path%11.cpp
        exit /b 1
    )
    set "host_cmd=host_temp.exe"
    set "host_ext=11.cpp"
    echo   Found: %host_path%11.cpp (C++11)
    goto :detect_my_player
)

echo ERROR: Host file not found: %host_path%
echo Supported formats: .py, .cpp
exit /b 1

:detect_my_player
REM Step 4: Detect my player programming language
echo Detecting my player: %my_player%...

set "my_cmd="
set "my_ext="

if exist "%my_player%.py" (
    set "my_cmd=python %my_player%.py"
    set "my_ext=.py"
    echo   Found: %my_player%.py (Python)
    goto :detect_opponent
)

if exist "%my_player%3.py" (
    set "my_cmd=python %my_player%3.py"
    set "my_ext=3.py"
    echo   Found: %my_player%3.py (Python3)
    goto :detect_opponent
)

if exist "%my_player%.cpp" (
    g++ -O2 %my_player%.cpp -o exe.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++ compilation failed for %my_player%.cpp
        exit /b 1
    )
    set "my_cmd=exe.exe"
    set "my_ext=.cpp"
    echo   Found: %my_player%.cpp (C++)
    goto :detect_opponent
)

if exist "%my_player%11.cpp" (
    g++ -std=c++11 -O2 %my_player%11.cpp -o exe.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++11 compilation failed for %my_player%11.cpp
        exit /b 1
    )
    set "my_cmd=exe.exe"
    set "my_ext=11.cpp"
    echo   Found: %my_player%11.cpp (C++11)
    goto :detect_opponent
)

if exist "%my_player%.java" (
    javac %my_player%.java 2>nul
    if errorlevel 1 (
        echo ERROR: Java compilation failed for %my_player%.java
        exit /b 1
    )
    set "my_cmd=java %my_player%"
    set "my_ext=.java"
    echo   Found: %my_player%.java (Java)
    goto :detect_opponent
)

echo ERROR: My player file not found: %my_player%
echo Supported formats: .py, .cpp, .java
exit /b 1

:detect_opponent
REM Step 5: Detect opponent player programming language
echo Detecting opponent player: %opponent_player%...

set "opp_cmd="
set "opp_ext="

if exist "%opponent_player%.py" (
    set "opp_cmd=python %opponent_player%.py"
    set "opp_ext=.py"
    echo   Found: %opponent_player%.py (Python)
    goto :start_game
)

if exist "%opponent_player%3.py" (
    set "opp_cmd=python %opponent_player%3.py"
    set "opp_ext=3.py"
    echo   Found: %opponent_player%3.py (Python3)
    goto :start_game
)

if exist "%opponent_player%.cpp" (
    g++ -O2 %opponent_player%.cpp -o exe_opponent.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++ compilation failed for %opponent_player%.cpp
        exit /b 1
    )
    set "opp_cmd=exe_opponent.exe"
    set "opp_ext=.cpp"
    echo   Found: %opponent_player%.cpp (C++)
    goto :start_game
)

if exist "%opponent_player%11.cpp" (
    g++ -std=c++11 -O2 %opponent_player%11.cpp -o exe_opponent.exe 2>nul
    if errorlevel 1 (
        echo ERROR: C++11 compilation failed for %opponent_player%11.cpp
        exit /b 1
    )
    set "opp_cmd=exe_opponent.exe"
    set "opp_ext=11.cpp"
    echo   Found: %opponent_player%11.cpp (C++11)
    goto :start_game
)

if exist "%opponent_player%.java" (
    javac %opponent_player%.java 2>nul
    if errorlevel 1 (
        echo ERROR: Java compilation failed for %opponent_player%.java
        exit /b 1
    )
    set "opp_cmd=java %opponent_player%"
    set "opp_ext=.java"
    echo   Found: %opponent_player%.java (Java)
    goto :start_game
)

echo ERROR: Opponent player file not found: %opponent_player%
echo Supported formats: .py, .cpp, .java
exit /b 1

:start_game
echo.

REM Step 6: Initialize result file
echo. > cresult.txt
echo %date% %time% >> cresult.txt

echo.
echo ============================================
echo Match Configuration:
echo   Host: %host_path%%host_ext%
echo   My Player: %my_player%%my_ext%
echo   Opponent:  %opponent_player%%opp_ext%
echo   Total rounds: %play_time%
echo ============================================
echo %date% %time%

echo. >> cresult.txt
echo ==Playing: %my_player% vs %opponent_player% with %host_path%== >> cresult.txt
echo Total rounds: %play_time% >> cresult.txt
echo %date% %time% >> cresult.txt

REM Initialize counters
set "black_win_time=0"
set "white_win_time=0"
set "black_tie=0"
set "white_tie=0"

REM Step 7: Game loop
set "round=1"

:round_loop

REM ============================================
REM Odd round: Opponent as Black, You as White
REM ============================================
echo.
echo =====Round %round%=====
echo =====Round %round%===== >> cresult.txt
echo Black:%opponent_player% White:%my_player%
echo Black:%opponent_player% White:%my_player% >> cresult.txt

call :play_game "%opp_cmd%" "%my_cmd%"

if "%game_result%"=="2" (
    echo White^(%my_player%^) win
    echo White^(%my_player%^) win >> cresult.txt
    set /a white_win_time+=1
) else if "%game_result%"=="0" (
    echo Tie.
    echo Tie. >> cresult.txt
    set /a white_tie+=1
) else (
    echo White^(%my_player%^) lose
    echo White^(%my_player%^) lose >> cresult.txt
)

REM ============================================
REM Even round: You as Black, Opponent as White
REM ============================================
set /a round+=1
echo.
echo =====Round %round%=====
echo =====Round %round%===== >> cresult.txt
echo Black:%my_player% White:%opponent_player%
echo Black:%my_player% White:%opponent_player% >> cresult.txt

call :play_game "%my_cmd%" "%opp_cmd%"

if "%game_result%"=="1" (
    echo Black^(%my_player%^) win
    echo Black^(%my_player%^) win >> cresult.txt
    set /a black_win_time+=1
) else if "%game_result%"=="0" (
    echo Tie.
    echo Tie. >> cresult.txt
    set /a black_tie+=1
) else (
    echo Black^(%my_player%^) lose
    echo Black^(%my_player%^) lose >> cresult.txt
)

REM Check if we've completed all rounds
set /a round+=1
if %round% LEQ %play_time% goto round_loop

REM ============================================
REM Summary
REM ============================================
set /a half_rounds=%play_time% / 2
set /a black_lose=%half_rounds% - black_win_time - black_tie
set /a white_lose=%half_rounds% - white_win_time - white_tie
set /a total_wins=black_win_time + white_win_time
set /a total_loses=black_lose + white_lose
set /a total_ties=black_tie + white_tie

echo.
echo =====Summary=====
echo Match: %my_player% vs %opponent_player% (Host: %host_path%)
echo Total rounds played: %play_time%
echo.
echo As Black Player ^| Win: %black_win_time% ^| Lose: %black_lose% ^| Tie: %black_tie%
echo As White Player ^| Win: %white_win_time% ^| Lose: %white_lose% ^| Tie: %white_tie%
echo.
echo Total: Win=%total_wins% Lose=%total_loses% Tie=%total_ties%

echo. >> cresult.txt
echo =====Summary===== >> cresult.txt
echo Match: %my_player% vs %opponent_player% (Host: %host_path%) >> cresult.txt
echo Total rounds played: %play_time% >> cresult.txt
echo As Black Player ^| Win: %black_win_time% ^| Lose: %black_lose% ^| Tie: %black_tie% >> cresult.txt
echo As White Player ^| Win: %white_win_time% ^| Lose: %white_lose% ^| Tie: %white_tie% >> cresult.txt
echo Total: Win=%total_wins% Lose=%total_loses% Tie=%total_ties% >> cresult.txt

REM Clean up
if exist "input.txt" del /f /q input.txt
if exist "output.txt" del /f /q output.txt
if exist "*.class" del /f /q *.class 2>nul
if exist "exe.exe" del /f /q exe.exe
if exist "exe_opponent.exe" del /f /q exe_opponent.exe
if exist "host_temp.exe" del /f /q host_temp.exe

echo.
echo Test completed.
echo Results saved to cresult.txt

goto :eof

REM ============================================
REM Subroutine: play_game
REM Parameters: %1 - Black command, %2 - White command
REM Returns: game_result (0=tie, 1=black win, 2=white win)
REM ============================================
:play_game

echo Clean up...
echo Clean up... >> cresult.txt

if exist "input.txt" del /f /q input.txt
if exist "output.txt" del /f /q output.txt

copy /y ".\init\input.txt" "input.txt" >nul

echo Start Playing...
echo Start Playing... >> cresult.txt

set "moves=0"

:game_loop

if exist "output.txt" del /f /q output.txt

echo Black makes move...
echo Black makes move... >> cresult.txt

%~1
set /a moves+=1

%host_cmd% -m %moves% -v True
if errorlevel 1 (
    set "game_result=%errorlevel%"
    goto :game_done
)

if exist "output.txt" del /f /q output.txt

echo White makes move...
echo White makes move... >> cresult.txt

%~2
set /a moves+=1

%host_cmd% -m %moves% -v True
if errorlevel 1 (
    set "game_result=%errorlevel%"
    goto :game_done
)

goto :game_loop

:game_done
goto :eof
