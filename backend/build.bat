@echo off
setlocal enabledelayedexpansion

REM 切换到脚本所在目录的上级，也就是项目根目录
cd /d %~dp0..
echo 当前工作目录：%cd%

REM 清理旧的构建
echo 清理旧的构建...
if exist backend\dist rmdir /s /q backend\dist
if exist backend\build rmdir /s /q backend\build
if exist BillNote_frontend\src-tauri\bin rmdir /s /q BillNote_frontend\src-tauri\bin
echo 清理完成。

REM 重新创建 Tauri 需要的目录结构
mkdir BillNote_frontend\src-tauri\bin

REM 获取 Rust 的 target triple（适配 Tauri 对应平台）
for /f "tokens=2 delims=:" %%A in ('rustc -Vv ^| findstr "host"') do (
    set "TARGET_TRIPLE=%%A"
)
set "TARGET_TRIPLE=%TARGET_TRIPLE: =%"
echo Detected target triple: %TARGET_TRIPLE%

if "%PYTHON_EXE%"=="" (
    set "PYTHON_EXE=python"
)
echo Using Python: %PYTHON_EXE%


REM --- 核心修改部分开始 ---

REM 步骤 1: 为了避免 PyInstaller 的解析歧义，我们先手动复制文件
echo 为打包准备 .env 文件...
copy .env.example backend\.env

REM 步骤 2: 执行 PyInstaller 打包，直接添加已存在的 .env 文件
echo 开始 PyInstaller 打包...
"%PYTHON_EXE%" -m PyInstaller ^
  -y ^
  --name VideoNoteBackend ^
  --paths backend ^
  --distpath BillNote_frontend\src-tauri\bin ^
  --workpath backend\build ^
  --specpath backend ^
  --hidden-import uvicorn ^
  --hidden-import fastapi ^
  --hidden-import starlette ^
  --collect-all numpy ^
  --collect-all chromadb ^
  --add-data "app\db\builtin_providers.json;." ^
  --add-data ".env;." ^
  backend\main.py

REM 步骤 3: 清理在项目根目录创建的临时 .env 文件
echo 清理临时的 .env 文件...
del backend\.env

REM --- 核心修改部分结束 ---


REM 重命名生成的可执行文件为符合 Tauri 要求的名称
move /Y BillNote_frontend\src-tauri\bin\VideoNoteBackend\VideoNoteBackend.exe BillNote_frontend\src-tauri\bin\VideoNoteBackend\VideoNoteBackend-%TARGET_TRIPLE%.exe

echo PyInstaller 打包完成：
dir BillNote_frontend\src-tauri\bin\VideoNoteBackend

echo 请检查 BillNote_frontend\src-tauri\bin\VideoNoteBackend 目录，确认其中包含了名为 .env 的【文件】。

endlocal
