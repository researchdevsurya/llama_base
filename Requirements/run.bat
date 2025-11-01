# Close all Python/VS Code terminals first!

echo "Removing all Python installations..."

# Uninstall Python via winget (if installed from Microsoft Store or installer)
winget uninstall "Python" -y
winget uninstall "Python 3" -y
winget uninstall "Python Launcher" -y

# Remove common install directories
Remove-Item -Recurse -Force "C:\Program Files\Python*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\Program Files (x86)\Python*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LocalAppData\Programs\Python" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:LocalAppData\Microsoft\WindowsApps\Python*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:AppData\Roaming\Python" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:AppData\Local\pip" -ErrorAction SilentlyContinue

# Clean environment PATH entries
$envPaths = [Environment]::GetEnvironmentVariable("Path", "User").Split(";") | Where-Object {$_ -notmatch "Python"}
[Environment]::SetEnvironmentVariable("Path", ($envPaths -join ";"), "User")

$sysPaths = [Environment]::GetEnvironmentVariable("Path", "Machine").Split(";") | Where-Object {$_ -notmatch "Python"}
[Environment]::SetEnvironmentVariable("Path", ($sysPaths -join ";"), "Machine")

echo "✅ All Python folders and PATH entries removed."
